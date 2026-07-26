import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np

# We must mock before importing main because it imports torch and sentence_transformers
with patch("continuum_embedding.main.SentenceTransformer"), patch("continuum_embedding.main.Prisma"):
    from continuum_embedding.main import run_worker

@pytest.mark.asyncio
async def test_worker_polling():
    with patch("continuum_embedding.main.Prisma") as MockPrisma, \
         patch("continuum_embedding.main.SentenceTransformer") as MockST:
         
        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.disconnect = AsyncMock()
        
        # Simulate returning 1 row on first poll, then 0 rows
        mock_db.query_raw = AsyncMock(side_effect=[
            [{"id": "doc1", "text": "Hello world"}],
            [] # Second poll returns empty to let us break or sleep
        ])
        mock_db.execute_raw = AsyncMock()
        MockPrisma.return_value = mock_db
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        MockST.return_value = mock_model
        
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt() # Break the infinite loop
            
            try:
                await run_worker()
            except KeyboardInterrupt:
                pass
                
        # Assertions
        mock_db.query_raw.assert_called()
        mock_model.encode.assert_called_with(["Hello world"], batch_size=1, convert_to_numpy=True)
        mock_db.execute_raw.assert_called_once()
        
        # Check the insert query arguments
        args, kwargs = mock_db.execute_raw.call_args
        query = args[0]
        assert "INSERT INTO embeddings" in query
        assert "doc1" in args # The document_id should be in the args
        assert "[0.1,0.2,0.3]" in args # The vector string should be in the args
