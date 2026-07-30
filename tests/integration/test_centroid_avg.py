"""AVG(vector) must equal the mean the drift worker used to compute in Python.

The worker previously selected every embedding in a window as text and averaged the parsed
arrays, so memory grew with the window. Moving the average into Postgres is only safe if it
produces the same centroid, and that is a claim about pgvector rather than about our code,
so it is tested against a real server.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.integration

DEFAULT_DSN = "postgresql://continuum:continuum@localhost:5432/continuum"


def connection():
    psycopg = pytest.importorskip("psycopg")
    dsn = os.getenv("CENTROID_TEST_DSN", DEFAULT_DSN)
    try:
        return psycopg.connect(dsn, connect_timeout=5)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no postgres at {dsn}: {exc}")


@pytest.fixture
def cursor():
    conn = connection()
    with conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS centroid_probe")
        cur.execute("CREATE TABLE centroid_probe (id serial primary key, v vector(8))")
        yield cur
        cur.execute("DROP TABLE IF EXISTS centroid_probe")
    conn.close()


def literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


def test_database_average_matches_numpy_mean(cursor):
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(200, 8)).astype(np.float64)
    for vector in vectors:
        cursor.execute("INSERT INTO centroid_probe (v) VALUES (%s::vector)", (literal(vector),))

    cursor.execute("SELECT AVG(v)::text, COUNT(*)::int FROM centroid_probe")
    rendered, count = cursor.fetchone()
    from_database = np.array([float(x) for x in rendered.strip("[]").split(",")])

    assert count == 200
    # pgvector accumulates in float4, so agreement is to single precision rather than exact.
    assert np.allclose(from_database, vectors.mean(axis=0), atol=1e-5)


def test_average_of_no_rows_is_null_rather_than_zero(cursor):
    """The worker treats a null centroid as "no window", not as the origin."""
    cursor.execute("SELECT AVG(v)::text, COUNT(*)::int FROM centroid_probe")
    rendered, count = cursor.fetchone()

    assert rendered is None
    assert count == 0


def test_average_is_returned_in_the_expected_text_shape(cursor):
    """The worker parses the rendering, so the format is part of the contract."""
    cursor.execute("INSERT INTO centroid_probe (v) VALUES (%s::vector)", (literal(np.ones(8)),))

    cursor.execute("SELECT AVG(v)::text FROM centroid_probe")
    rendered = cursor.fetchone()[0]

    assert rendered.startswith("[") and rendered.endswith("]")
    assert len(rendered.strip("[]").split(",")) == 8
