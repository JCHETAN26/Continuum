import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone, timedelta
import numpy as np

# Mock dependencies before import
with patch("continuum_drift.worker.Prisma"):
    from continuum_drift.worker import compute_centroid, process_window

@pytest.mark.asyncio
async def test_compute_centroid():
    mock_db = MagicMock()
    mock_db.query_raw = AsyncMock(return_value=[
        {"vec_str": "[1.0, 2.0, 3.0]"},
        {"vec_str": "[3.0, 2.0, 1.0]"}
    ])
    
    start = datetime.now(timezone.utc)
    end = start + timedelta(minutes=5)
    
    centroid = await compute_centroid(mock_db, start, end)
    
    assert centroid is not None
    assert np.allclose(centroid, np.array([2.0, 2.0, 2.0]))

@pytest.mark.asyncio
async def test_process_window_creates_alert(monkeypatch):
    mock_db = MagicMock()
    
    # Simulate existing window = None
    mock_db.driftwindow = MagicMock()
    mock_db.driftwindow.find_unique = AsyncMock(return_value=None)
    
    # Baseline id and centroid
    async def mock_get_baseline(*args, **kwargs):
        return ("baseline_id", np.array([1.0, 1.0, 1.0]))
        
    monkeypatch.setattr("continuum_drift.worker.get_or_create_baseline", mock_get_baseline)
    
    # Compute centroid returns a very different vector
    async def mock_compute(*args, **kwargs):
        return np.array([-1.0, -1.0, -1.0])
        
    monkeypatch.setattr("continuum_drift.worker.compute_centroid", mock_compute)
    
    mock_db.execute_raw = AsyncMock()
    
    mock_producer = MagicMock()
    
    await process_window(mock_db, mock_producer, "FIVE_MIN", timedelta(minutes=5))
    
    # Should insert into drift_windows
    mock_db.execute_raw.assert_called_once()
    
    # Should produce an alert since distance is high
    mock_producer.produce.assert_called_once()
    
    args, kwargs = mock_producer.produce.call_args
    assert args[0] == "drift-alerts"
    assert b"FIVE_MIN" in kwargs["key"]
    assert b"drift_score" in kwargs["value"]
