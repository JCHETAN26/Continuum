from typing import Any, Literal

from pydantic import AnyUrl, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Runtime
    node_env: Literal["development", "production", "test"] = "development"
    log_level: Literal["fatal", "error", "warn", "info", "debug", "trace", "silent"] = "info"

    # Postgres (pgvector)
    postgres_user: str = Field(min_length=1)
    postgres_password: str = Field(min_length=1)
    postgres_db: str = Field(min_length=1)
    postgres_port: int = Field(gt=0)
    database_url: PostgresDsn
    shadow_database_url: PostgresDsn | None = None

    # Kafka (Redpanda)
    kafka_brokers: str = Field(min_length=1)
    kafka_client_id: str = Field(min_length=1)
    redpanda_kafka_port: int = Field(gt=0)
    redpanda_admin_port: int = Field(gt=0)
    redpanda_console_port: int = Field(gt=0)

    # Redis
    redis_url: RedisDsn
    redis_port: int = Field(gt=0)

    # Object Storage (MinIO)
    minio_root_user: str = Field(min_length=1)
    minio_root_password: str = Field(min_length=1)
    minio_api_port: int = Field(gt=0)
    minio_console_port: int = Field(gt=0)

    s3_endpoint: AnyUrl
    s3_region: str = Field(min_length=1)
    s3_access_key_id: str = Field(min_length=1)
    s3_secret_access_key: str = Field(min_length=1)
    s3_bucket_documents: str = Field(min_length=1)
    s3_bucket_models: str = Field(min_length=1)
    s3_force_path_style: bool

    # Embedding model
    embedding_model: str = Field(min_length=1)
    embedding_dim: int = Field(gt=0)

    # Drift detection
    drift_threshold: float = Field(ge=0.0, le=1.0)
    linguistic_drift_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    linguistic_drift_window_minutes: int = Field(default=2, gt=0)
    linguistic_drift_poll_seconds: int = Field(default=30, gt=0)
    drift_trigger_min_documents: int = Field(default=100, ge=0)
    # Same scale as drift_threshold: cosine distance between window centroids, where a
    # domain shift reads around 0.10. A value tuned for a normalised 0-1 drift score
    # (0.75) is unreachable here and silently disables embedding-triggered retraining.
    drift_trigger_min_embedding_drift: float = Field(default=0.08, ge=0.0, le=1.0)
    drift_trigger_min_linguistic_drift: float = Field(default=0.60, ge=0.0, le=1.0)
    drift_trigger_cooldown_hours: float = Field(default=6.0, ge=0.0)
    drift_trigger_max_daily_trains: int = Field(default=3, ge=0)
    # Fractional retrieval-quality gain a candidate must show over the active model before
    # it is eligible for activation. Held well above measurement noise: an adapter that
    # merely reshuffles near-tied neighbours can move quality by a fraction of a percent
    # without retrieving anything better.
    activation_min_improvement: float = Field(default=0.10, ge=0.0, le=1.0)

    # Trainer
    trainer_backend: Literal["demo_adapter", "peft"] = "demo_adapter"
    retrain_cooldown_minutes: int = Field(default=10, ge=0)
    training_retry_base_seconds: float = Field(default=1.0, ge=0.0)

    # Observability
    otel_service_name: str = Field(min_length=1)
    otel_exporter_otlp_endpoint: AnyUrl | None = None

    # Operations
    retention_cleanup_interval_seconds: int = Field(default=86400, gt=0)
    retention_embeddings_days: int = Field(default=90, gt=0)
    retention_drift_windows_days: int = Field(default=30, gt=0)
    retention_training_jobs_days: int = Field(default=365, gt=0)
    rollback_error_rate_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    rollback_window_seconds: int = Field(default=300, gt=0)
    rollback_min_requests: int = Field(default=100, gt=0)

    @field_validator("otel_exporter_otlp_endpoint", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
