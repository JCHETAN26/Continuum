import importlib.util
import sys
from pathlib import Path


def test_ops_migration_is_reversible() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "shared"
        / "alembic"
        / "versions"
        / "20260728_0001_ops_runtime_tables.py"
    )
    spec = importlib.util.spec_from_file_location("ops_runtime_tables", migration_path)
    assert spec and spec.loader
    migration = importlib.util.module_from_spec(spec)
    sys.modules["ops_runtime_tables"] = migration
    spec.loader.exec_module(migration)

    assert migration.revision == "20260728_0001"
    assert callable(migration.upgrade)
    assert callable(migration.downgrade)
