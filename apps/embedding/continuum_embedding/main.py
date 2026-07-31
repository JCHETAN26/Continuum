"""Embedding worker.

Claims documents that have no vector, and documents whose vector was produced by a model
version other than the one currently ACTIVE. The second case is what makes activating an
adapted encoder safe: vectors from two different encoders are not comparable, so the whole
corpus has to be re-encoded, and doing it here rather than inside the training pipeline
means the work is resumable, survives a restart, and spreads across replicas.

Claiming uses FOR UPDATE ... SKIP LOCKED, so replicas take disjoint batches without
coordinating.
"""

import asyncio
import uuid

import structlog
from continuum_shared.active_model import LoadedModel, load_active_model
from continuum_shared.config import settings
from continuum_shared.embeddings import (
    embed_texts,
    encode_with_session,
    get_tokenizer,
    vector_literal,
)
from continuum_shared.prisma import Prisma

logger = structlog.get_logger()

BATCH_SIZE = 50
IDLE_SLEEP_SECONDS = 1.0

# Documents with no vector, or whose vector predates the active model. IS DISTINCT FROM
# treats a null model_version_id as "not the active model", so vectors written before this
# column was populated are re-encoded once and then settle.
CLAIM_QUERY = """
    SELECT d.id, d.text
    FROM documents d
    LEFT JOIN embeddings e ON e.document_id = d.id
    WHERE e.id IS NULL
       OR e.model_version_id IS DISTINCT FROM $1::uuid
    LIMIT $2
    FOR UPDATE OF d SKIP LOCKED
"""

# Same claim without an active model to compare against: only documents lacking a vector.
CLAIM_UNEMBEDDED_QUERY = """
    SELECT d.id, d.text
    FROM documents d
    WHERE NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.document_id = d.id)
    LIMIT $1
    FOR UPDATE OF d SKIP LOCKED
"""

# first_embedded_at is written on insert and deliberately absent from the update clause:
# it records when a document first became searchable, and drift windows are built from it.
# created_at continues to move on every write, which is what makes it useful for watching
# a backfill progress.
UPSERT_QUERY = """
    INSERT INTO embeddings (
        id, document_id, model_version_id, vector, dimension, created_at, first_embedded_at
    )
    VALUES ($1::uuid, $2::uuid, $3::uuid, $4::vector, $5, NOW(), NOW())
    ON CONFLICT (document_id) DO UPDATE
    SET vector = EXCLUDED.vector,
        model_version_id = EXCLUDED.model_version_id,
        dimension = EXCLUDED.dimension,
        created_at = NOW()
"""


def encode(texts: list[str], model: LoadedModel | None) -> list[list[float]]:
    """Embed with the active encoder when there is one, otherwise the base model."""
    if model is not None and model.is_encoder:
        vectors = encode_with_session(model.session, get_tokenizer(), texts)
        return [[float(value) for value in row] for row in vectors]
    return embed_texts(texts, settings.embedding_dim)


async def resolve_model(db: Prisma, current: LoadedModel | None) -> LoadedModel | None:
    """Reload the artifact only when the ACTIVE version changes."""
    try:
        latest = await load_active_model(db)
    except Exception as error:
        # A malformed or unreachable artifact must not stop the worker embedding new
        # documents with the base model.
        logger.warning(
            "Could not load the active model, continuing on the base model", error=str(error)
        )
        return current

    if latest is None:
        return None
    if current is not None and latest.version == current.version:
        return current

    logger.info(
        "Embedding against a new active model",
        version=latest.version,
        kind=latest.kind,
        encoder=latest.is_encoder,
    )
    return latest


async def run_worker() -> None:
    db = Prisma()
    await db.connect()
    logger.info("Worker started", dimension=settings.embedding_dim, batch_size=BATCH_SIZE)

    model: LoadedModel | None = None
    try:
        while True:
            model = await resolve_model(db, model)

            if model is None:
                rows = await db.query_raw(CLAIM_UNEMBEDDED_QUERY, BATCH_SIZE)
            else:
                rows = await db.query_raw(CLAIM_QUERY, model.model_id, BATCH_SIZE)

            if not rows:
                await asyncio.sleep(IDLE_SLEEP_SECONDS)
                continue

            texts = [row["text"] for row in rows]
            vectors = encode(texts, model)

            for row, vector in zip(rows, vectors, strict=True):
                await db.execute_raw(
                    UPSERT_QUERY,
                    str(uuid.uuid4()),
                    row["id"],
                    model.model_id if model else None,
                    vector_literal(vector),
                    settings.embedding_dim,
                )

            logger.info(
                "Embedded batch",
                documents=len(rows),
                model_version=model.version if model else "base",
            )

    except KeyboardInterrupt:
        pass
    finally:
        await db.disconnect()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
