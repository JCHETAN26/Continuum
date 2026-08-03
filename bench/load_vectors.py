"""Load-test the retrieval data plane at a corpus size the demo never reaches.

The end-to-end demo runs on 1,200 documents. That is enough to show the adaptation loop
working and far too small to say anything about scale, so this loads synthetic vectors
directly and measures the paths that corpus size actually stresses:

- the claim query the embedding worker polls to find work
- the drift centroid aggregate
- approximate nearest-neighbour search through the HNSW index
- the UPDATE throughput of a re-index

**The vectors are random and mean nothing.** Retrieval quality cannot be measured this way,
and this script does not try. What it measures is the storage and query plane, which is
identical whether a vector came from MiniLM or from a random number generator. Embedding a
million documents for real would take days on CPU at the measured 3.1 docs/sec per replica;
that cost is orthogonal to whether Postgres can serve the queries.

Run against a throwaway database. It creates the real product schema from the Prisma
migrations, so the indexes under test are the ones production uses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psycopg

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = REPO_ROOT / "packages" / "shared" / "prisma" / "migrations"
DIMENSION = 384
COPY_CHUNK = 5_000


@dataclass
class Measurement:
    name: str
    rows: int
    milliseconds: float
    detail: str = ""

    def render(self) -> str:
        return f"{self.name:<34} rows={self.rows:<9} {self.milliseconds:9.1f} ms  {self.detail}"


def migration_files() -> list[Path]:
    """Every migration, in the order Postgres applies them."""
    return sorted(MIGRATIONS.glob("*/migration.sql"), key=lambda path: path.parent.name)


def apply_schema(connection: psycopg.Connection) -> None:
    for path in migration_files():
        connection.execute(path.read_text(encoding="utf-8"))
    connection.commit()


def load_documents(connection: psycopg.Connection, count: int, seed: int) -> float:
    """Insert documents and their vectors, in chunks, with the HNSW index live.

    The index is left in place rather than dropped and rebuilt, because that is the state
    an ingest stream actually writes into.
    """
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    for offset in range(0, count, COPY_CHUNK):
        chunk = min(COPY_CHUNK, count - offset)
        vectors = rng.normal(size=(chunk, DIMENSION)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        with connection.cursor() as cursor:
            with cursor.copy(
                "COPY documents (external_id, idempotency_key, text, source, occurred_at, "
                "content_hash) FROM STDIN"
            ) as copy:
                for index in range(chunk):
                    row = offset + index
                    copy.write_row(
                        (
                            f"load-{row}",
                            f"load-key-{row}",
                            f"synthetic load-test document {row}",
                            "load_test",
                            "2026-01-01T00:00:00+00:00",
                            # documents_content_hash_check enforces ^[0-9a-f]{64}$.
                            f"{row:064x}",
                        )
                    )
        connection.commit()

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM documents WHERE external_id = ANY(%s) ORDER BY external_id",
                ([f"load-{offset + index}" for index in range(chunk)],),
            )
            document_ids = [row[0] for row in cursor.fetchall()]

            with cursor.copy(
                "COPY embeddings (document_id, vector, dimension, created_at, "
                "first_embedded_at) FROM STDIN"
            ) as copy:
                for document_id, vector in zip(document_ids, vectors, strict=False):
                    literal = "[" + ",".join(f"{value:.5f}" for value in vector) + "]"
                    copy.write_row(
                        (
                            document_id,
                            literal,
                            DIMENSION,
                            "2026-01-01T00:00:00+00:00",
                            "2026-01-01T00:00:00+00:00",
                        )
                    )
        connection.commit()

    return (time.perf_counter() - started) * 1000.0


Params = tuple | dict | None


def time_query(
    connection: psycopg.Connection, sql: str, params: Params = (), runs: int = 3
) -> float:
    """Median wall-clock of a query, so one cold run does not define the number."""
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            cursor.fetchall()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def explain(connection: psycopg.Connection, sql: str, params: Params = ()) -> str:
    """Access method plus buffers touched.

    The access method alone is not enough to tell a healthy plan from a sick one: every
    query here ends in LIMIT 50, so a sequential scan that finds fifty matching rows and
    stops is cheap, while one that reads the whole table to prove nothing matches is the
    problem. Buffers separate the two, and are the same number regardless of how loaded the
    runner is, which wall-clock on shared CI is not.
    """
    with connection.cursor() as cursor:
        cursor.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, params)
        plan = " ".join(line[0] for line in cursor.fetchall())

    buffers = sum(int(count) for count in re.findall(r"shared hit=(\d+)", plan))
    if "Seq Scan" in plan:
        method = "SEQ SCAN"
    elif "Index" in plan:
        method = "index"
    else:
        method = "?"
    return f"{method}, {buffers} buffers"


# The two queries the embedding worker polls, mirroring apps/embedding/continuum_embedding/main.py.
#
# They were one query joined by OR until this benchmark reported it as a SEQ SCAN at 100k
# rows: `IS DISTINCT FROM` behind an OR with a null check has no btree strategy, so Postgres
# hashed both tables and filtered. Kept split here so the numbers below describe what the
# worker runs rather than a shape nobody executes.
CLAIM_UNEMBEDDED_QUERY = """
    SELECT d.id, d.text
    FROM documents d
    WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.document_id = d.id)
    LIMIT 50
"""

# IS NULL plus two range bounds is the same predicate as IS DISTINCT FROM, spelled so the
# planner can answer it from embeddings(model_version_id) with a BitmapOr.
CLAIM_STALE_QUERY = """
    SELECT d.id, d.text
    FROM embeddings e
    JOIN documents d ON d.id = e.document_id
    WHERE e.model_version_id IS NULL
       OR e.model_version_id < %(version)s::uuid
       OR e.model_version_id > %(version)s::uuid
    LIMIT 50
"""

DRIFT_CENTROID = """
    SELECT AVG(vector)::text, COUNT(*)::int
    FROM embeddings
    WHERE first_embedded_at >= %s::timestamptz AND first_embedded_at < %s::timestamptz
"""

ANN_SEARCH = """
    SELECT document_id FROM embeddings ORDER BY vector <=> %s::vector LIMIT 10
"""


def measure(connection: psycopg.Connection, rows: int, seed: int) -> list[Measurement]:
    rng = np.random.default_rng(seed + 1)
    probe = rng.normal(size=DIMENSION).astype(np.float32)
    probe /= np.linalg.norm(probe)
    probe_literal = "[" + ",".join(f"{value:.5f}" for value in probe) + "]"
    absent_version = "00000000-0000-0000-0000-000000000000"
    settled_version = "11111111-1111-1111-1111-111111111111"

    # With every row still unversioned the claim matches immediately and LIMIT 50 returns
    # after fifty rows, which is the backlog case and the easy one.
    results = [
        Measurement(
            "claim query, stale branch (backlog)",
            rows,
            time_query(connection, CLAIM_STALE_QUERY, {"version": absent_version}),
            explain(connection, CLAIM_STALE_QUERY, {"version": absent_version}),
        ),
        Measurement(
            "drift centroid over window",
            rows,
            time_query(
                connection,
                DRIFT_CENTROID,
                ("2025-12-31T00:00:00+00:00", "2026-01-02T00:00:00+00:00"),
            ),
        ),
        Measurement(
            "ANN search (HNSW, top 10)",
            rows,
            time_query(connection, ANN_SEARCH, (probe_literal,)),
        ),
    ]

    # The case that actually matters: every vector already carries the active version, so
    # the claim matches nothing and LIMIT cannot stop early. This is what the worker polls
    # once a backfill has drained, several times a second, forever.
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO model_versions (id, version, base_model, updated_at) "
            "VALUES (%s::uuid, 'load-test', 'synthetic', NOW()) ON CONFLICT DO NOTHING",
            (settled_version,),
        )
        cursor.execute("UPDATE embeddings SET model_version_id = %s::uuid", (settled_version,))
    connection.commit()

    results.extend(
        [
            Measurement(
                "claim query, stale branch (settled)",
                rows,
                time_query(connection, CLAIM_STALE_QUERY, {"version": settled_version}),
                explain(connection, CLAIM_STALE_QUERY, {"version": settled_version}),
            ),
            # The unembedded branch has no version to match on, so it stays an anti-join and
            # stays linear: proving no document is missing a vector means visiting them all.
            # Reported separately rather than folded in, because the two branches degrade for
            # different reasons and only one of them was fixable with an index.
            Measurement(
                "claim query, unembedded branch (settled)",
                rows,
                time_query(connection, CLAIM_UNEMBEDDED_QUERY, None),
                explain(connection, CLAIM_UNEMBEDDED_QUERY, None),
            ),
        ]
    )

    # Re-index throughput: what activating a model costs. Bounded to a slice so the
    # measurement stays proportional rather than rewriting the whole corpus.
    slice_size = min(5_000, rows)
    started = time.perf_counter()
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE embeddings SET vector = vector, created_at = NOW() "
            "WHERE document_id IN (SELECT id FROM documents LIMIT %s)",
            (slice_size,),
        )
    connection.commit()
    elapsed = (time.perf_counter() - started) * 1000.0
    results.append(
        Measurement(
            "re-index UPDATE",
            slice_size,
            elapsed,
            f"{slice_size / (elapsed / 1000.0):,.0f} rows/sec",
        )
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--dsn", default=os.environ.get("LOAD_TEST_DSN"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dsn:
        raise SystemExit("set --dsn or LOAD_TEST_DSN to a throwaway database")

    random.seed(args.seed)
    with psycopg.connect(args.dsn, autocommit=False) as connection:
        apply_schema(connection)
        load_ms = load_documents(connection, args.count, args.seed)
        print(f"loaded {args.count:,} vectors in {load_ms / 1000.0:,.1f}s")

        results = measure(connection, args.count, args.seed)
        for result in results:
            print("  " + result.render())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "rows": args.count,
                    "load_ms": load_ms,
                    "measurements": [asdict(r) for r in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
