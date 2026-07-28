import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "retention_cleanup.py"
spec = importlib.util.spec_from_file_location("retention_cleanup", MODULE_PATH)
assert spec and spec.loader
retention_cleanup = importlib.util.module_from_spec(spec)
sys.modules["retention_cleanup"] = retention_cleanup
spec.loader.exec_module(retention_cleanup)

RetentionPolicy = retention_cleanup.RetentionPolicy
run_retention_cleanup = retention_cleanup.run_retention_cleanup


class FakePrisma:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def execute_raw(self, sql: str, days: int) -> int:
        self.calls.append((sql, days))
        return len(self.calls)


@pytest.mark.asyncio
async def test_retention_cleanup_deletes_expected_tables() -> None:
    db = FakePrisma()

    result = await run_retention_cleanup(
        db,  # type: ignore[arg-type]
        RetentionPolicy(embeddings_days=90, drift_windows_days=30, training_jobs_days=365),
    )

    assert result == {
        "embeddings": 1,
        "drift_windows": 2,
        "linguistic_windows": 3,
        "training_jobs": 4,
    }
    assert [days for _, days in db.calls] == [90, 30, 30, 365]
    assert "model_versions" not in " ".join(sql for sql, _ in db.calls)
