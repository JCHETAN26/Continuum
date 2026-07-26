from typing import Literal

from pydantic import AnyUrl, Field, PostgresDsn, RedisDsn
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

    # Observability
    otel_service_name: str = Field(min_length=1)
    otel_exporter_otlp_endpoint: AnyUrl | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
