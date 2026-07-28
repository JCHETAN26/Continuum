from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision = "20260728_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            key_hash varchar(128) NOT NULL,
            label varchar(128) NOT NULL,
            created_at timestamptz(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
            revoked_at timestamptz(6)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS model_request_metrics (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            model_version varchar(64) NOT NULL,
            status_code integer NOT NULL,
            latency_ms double precision NOT NULL,
            observed_at timestamptz(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS model_rollbacks (
            id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            failed_version varchar(64) NOT NULL,
            restored_version varchar(64) NOT NULL,
            error_rate double precision NOT NULL,
            request_count integer NOT NULL,
            created_at timestamptz(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS api_keys_revoked_at_idx ON api_keys(revoked_at)")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS model_request_metrics_version_observed_idx
        ON model_request_metrics(model_version, observed_at)
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS model_rollbacks_created_at_idx ON model_rollbacks(created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS model_rollbacks_created_at_idx")
    op.execute("DROP INDEX IF EXISTS model_request_metrics_version_observed_idx")
    op.execute("DROP INDEX IF EXISTS api_keys_revoked_at_idx")
    op.execute("DROP TABLE IF EXISTS model_rollbacks")
    op.execute("DROP TABLE IF EXISTS model_request_metrics")
    op.execute("DROP TABLE IF EXISTS api_keys")
