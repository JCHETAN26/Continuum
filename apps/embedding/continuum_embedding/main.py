import asyncio
import uuid

import structlog
from continuum_shared.config import settings
from continuum_shared.embeddings import embed_texts, vector_literal
from continuum_shared.prisma import Prisma

logger = structlog.get_logger()


async def run_worker() -> None:
    db = Prisma()
    await db.connect()

    logger.info(
        "Worker started, polling for unembedded documents", dimension=settings.embedding_dim
    )

    try:
        while True:
            # Poll for 50 documents without embeddings
            query = """
                SELECT d.id, d.text 
                FROM documents d
                WHERE NOT EXISTS (
                    SELECT 1 FROM embeddings e WHERE e.document_id = d.id
                )
                LIMIT 50 
                FOR UPDATE OF d SKIP LOCKED;
            """
            rows = await db.query_raw(query)

            if not rows:
                await asyncio.sleep(1.0)
                continue

            logger.info(f"Processing batch of {len(rows)} documents")

            texts = [row["text"] for row in rows]
            doc_ids = [row["id"] for row in rows]

            embeddings = embed_texts(texts, settings.embedding_dim)

            # Insert embeddings
            for doc_id, emb in zip(doc_ids, embeddings):
                emb_id = str(uuid.uuid4())

                insert_query = """
                    INSERT INTO embeddings (id, document_id, vector, dimension, created_at)
                    VALUES ($1::uuid, $2::uuid, $3::vector, $4, NOW())
                    ON CONFLICT (document_id) DO NOTHING;
                """
                await db.execute_raw(
                    insert_query,
                    emb_id,
                    doc_id,
                    vector_literal(emb),
                    settings.embedding_dim,
                )

            logger.info(f"Successfully embedded and stored {len(rows)} documents")

    except KeyboardInterrupt:
        pass
    finally:
        await db.disconnect()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
