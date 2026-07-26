import asyncio
import uuid

import structlog
import torch
from continuum_shared.config import settings
from continuum_shared.prisma import Prisma
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

async def run_worker() -> None:
    db = Prisma()
    await db.connect()
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        
    logger.info(
        "Loading sentence-transformers model", 
        model=settings.embedding_model, 
        device=device
    )
    model = SentenceTransformer(settings.embedding_model, device=device)
    
    logger.info("Worker started, polling for unembedded documents")

    try:
        while True:
            # Poll for 50 documents without embeddings
            query = """
                SELECT d.id, d.text 
                FROM documents d 
                LEFT JOIN embeddings e ON d.id = e.document_id 
                WHERE e.document_id IS NULL 
                LIMIT 50 
                FOR UPDATE SKIP LOCKED;
            """
            rows = await db.query_raw(query)
            
            if not rows:
                await asyncio.sleep(1.0)
                continue
                
            logger.info(f"Processing batch of {len(rows)} documents")
            
            texts = [row['text'] for row in rows]
            doc_ids = [row['id'] for row in rows]
            
            # Compute embeddings
            embeddings = model.encode(texts, batch_size=len(texts), convert_to_numpy=True)
            
            # Insert embeddings
            for doc_id, emb in zip(doc_ids, embeddings):
                # Convert numpy array to pgvector string format: '[0.1, 0.2, ...]'
                emb_list = emb.tolist()
                emb_str = f"[{','.join(map(str, emb_list))}]"
                
                emb_id = str(uuid.uuid4())
                
                insert_query = """
                    INSERT INTO embeddings (id, document_id, vector, dimension, created_at)
                    VALUES ($1::uuid, $2::uuid, $3::vector, $4, NOW())
                    ON CONFLICT (document_id) DO NOTHING;
                """
                await db.execute_raw(insert_query, emb_id, doc_id, emb_str, settings.embedding_dim)
            
            logger.info(f"Successfully embedded and stored {len(rows)} documents")
            
    except KeyboardInterrupt:
        pass
    finally:
        await db.disconnect()

def main() -> None:
    asyncio.run(run_worker())

if __name__ == "__main__":
    main()
