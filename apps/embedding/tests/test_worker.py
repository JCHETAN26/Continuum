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

        # The worker resolves the active model before each claim, so the first call is
        # that lookup. With no ACTIVE row it embeds with the base model and claims only
        # documents that have no vector at all.
        mock_db.query_raw = AsyncMock(
            side_effect=[
                [],  # no ACTIVE model version
                [{"id": "doc1", "text": "Hello world"}],
                [],  # active-model lookup on the second cycle
                [],  # nothing left to claim
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

        # Check the upsert arguments
        args, kwargs = mock_db.execute_raw.call_args
        query = args[0]
        assert "INSERT INTO embeddings" in query
        assert "doc1" in args  # The document_id should be in the args
        # No ACTIVE model, so the vector is written without a model version stamp.
        assert args[3] is None
        vector_arg = args[4]
        assert vector_arg.startswith("[")
        assert vector_arg.endswith("]")
