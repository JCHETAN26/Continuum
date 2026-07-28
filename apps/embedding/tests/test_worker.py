from unittest.mock import AsyncMock, MagicMock, patch

import pytest

with patch("continuum_embedding.main.Prisma"):
    from continuum_embedding.main import run_worker


@pytest.mark.asyncio
async def test_worker_polling():
    with patch("continuum_embedding.main.Prisma") as mock_prisma_cls:
        mock_db = MagicMock()
        mock_db.connect = AsyncMock()
        mock_db.disconnect = AsyncMock()

        # Simulate returning 1 row on first poll, then 0 rows
        mock_db.query_raw = AsyncMock(
            side_effect=[
                [{"id": "doc1", "text": "Hello world"}],
                [],  # Second poll returns empty to let us break or sleep
            ]
        )
        mock_db.execute_raw = AsyncMock()
        mock_prisma_cls.return_value = mock_db

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()  # Break the infinite loop

            try:
                await run_worker()
            except KeyboardInterrupt:
                pass

        # Assertions
        mock_db.query_raw.assert_called()
        mock_db.execute_raw.assert_called_once()

        # Check the insert query arguments
        args, kwargs = mock_db.execute_raw.call_args
        query = args[0]
        assert "INSERT INTO embeddings" in query
        assert "doc1" in args  # The document_id should be in the args
        vector_arg = args[3]
        assert vector_arg.startswith("[")
        assert vector_arg.endswith("]")
