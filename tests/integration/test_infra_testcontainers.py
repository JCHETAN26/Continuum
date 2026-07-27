import os
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_DOCKER_INTEGRATION") != "1",
        reason="Set RUN_DOCKER_INTEGRATION=1 to start Docker-backed Testcontainers.",
    ),
]


def _container_host_port(container, port: int) -> tuple[str, int]:
    return container.get_container_host_ip(), int(container.get_exposed_port(port))


def _wait_for_postgres_ready(container, user: str, database: str, timeout: int = 60) -> None:
    """Poll pg_isready until the post-init server accepts connections.

    The postgres entrypoint logs "ready to accept connections" twice: once for the
    temporary socket-only server that runs /docker-entrypoint-initdb.d, and again
    after it restarts for real. Waiting on that log line races the restart, so poll
    the server itself instead.
    """
    deadline = time.time() + timeout
    last_output = b""
    while time.time() < deadline:
        result = container.exec(["pg_isready", "-U", user, "-d", database])
        if result.exit_code == 0:
            return
        last_output = result.output
        time.sleep(1)

    raise AssertionError(f"postgres not ready within {timeout}s: {last_output.decode()}")


def test_postgres_pgvector_migration_accepts_embeddings():
    from pathlib import Path

    from testcontainers.core.container import DockerContainer

    migration = Path("packages/shared/prisma/migrations/20260726000000_init/migration.sql")
    with (
        DockerContainer("pgvector/pgvector:pg16")
        .with_env("POSTGRES_USER", "continuum")
        .with_env("POSTGRES_PASSWORD", "continuum")
        .with_env("POSTGRES_DB", "continuum")
        .with_volume_mapping(str(migration.resolve()), "/docker-entrypoint-initdb.d/01-schema.sql")
        .with_exposed_ports(5432)
    ) as postgres:
        _wait_for_postgres_ready(postgres, user="continuum", database="continuum")

        vector = "[" + ",".join(["0.001"] * 384) + "]"
        insert_sql = (
            "INSERT INTO documents "
            "(id, external_id, idempotency_key, text, source, occurred_at, content_hash) "
            "VALUES "
            f"('{uuid.uuid4()}', 'doc-1', 'idem-1', 'hello', 'integration', now(), "
            "'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');"
            "INSERT INTO embeddings (id, document_id, vector, dimension) "
            "SELECT gen_random_uuid(), id, "
            f"'{vector}'::vector, 384 FROM documents WHERE external_id = 'doc-1';"
            "SELECT COUNT(*) FROM embeddings;"
        )
        result = postgres.exec(
            [
                "psql",
                "-U",
                "continuum",
                "-d",
                "continuum",
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                insert_sql,
            ]
        )

        assert result.exit_code == 0
        assert "1" in result.output.decode()


def test_redis_roundtrip_with_real_container():
    from redis import Redis
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    with DockerContainer("redis:7.4-alpine").with_exposed_ports(6379) as redis_container:
        wait_for_logs(redis_container, "Ready to accept connections", timeout=20)
        host, port = _container_host_port(redis_container, 6379)

        client = Redis(host=host, port=port, decode_responses=True)
        assert client.ping() is True
        assert client.set("continuum:integration", "ok") is True
        assert client.get("continuum:integration") == "ok"


def test_minio_bucket_object_roundtrip_with_real_container():
    from io import BytesIO

    from minio import Minio
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    access_key = "continuum"
    secret_key = "continuum-local-dev"

    with (
        DockerContainer("minio/minio:RELEASE.2024-12-18T13-15-44Z")
        .with_command('server /data --console-address ":9001"')
        .with_env("MINIO_ROOT_USER", access_key)
        .with_env("MINIO_ROOT_PASSWORD", secret_key)
        .with_exposed_ports(9000)
    ) as minio_container:
        wait_for_logs(minio_container, "API:", timeout=30)
        host, port = _container_host_port(minio_container, 9000)

        client = Minio(
            f"{host}:{port}",
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        bucket = "continuum-models"
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        body = b"onnx-bytes"
        client.put_object(bucket, "integration/model.onnx", BytesIO(body), length=len(body))
        response = client.get_object(bucket, "integration/model.onnx")
        try:
            assert response.read() == body
        finally:
            response.close()
            response.release_conn()


def test_redpanda_produce_consume_with_real_container():
    from confluent_kafka import Consumer, Producer
    from testcontainers.community.kafka import RedpandaContainer

    class PortStableRedpandaContainer(RedpandaContainer):
        """Tolerates Docker publishing the port map slightly after the container starts.

        RedpandaContainer.start() looks up the mapped 9092 immediately, to build the
        advertised OUTSIDE listener address. On Docker Desktop that lookup
        intermittently lands before the mapping exists and raises ConnectionError.
        """

        def get_exposed_port(self, port: int, timeout: float = 30.0) -> int:
            deadline = time.time() + timeout
            while True:
                try:
                    return super().get_exposed_port(port)
                except ConnectionError:
                    if time.time() >= deadline:
                        raise
                    time.sleep(0.25)

    # RedpandaContainer advertises a separate OUTSIDE listener bound to the mapped host
    # port. A single-listener container can only advertise the in-container port, so
    # clients follow the broker metadata to an unreachable address and time out.
    with PortStableRedpandaContainer("redpandadata/redpanda:v24.3.4") as redpanda:
        bootstrap = redpanda.get_bootstrap_server()
        topic = f"continuum-integration-{uuid.uuid4()}"

        producer = Producer({"bootstrap.servers": bootstrap, "client.id": "continuum-test"})
        producer.produce(topic, key=b"demo", value=b"ready")
        producer.flush(10)

        consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": f"continuum-test-{uuid.uuid4()}",
                "auto.offset.reset": "earliest",
            }
        )
        consumer.subscribe([topic])
        try:
            deadline = time.time() + 20
            message = None
            while time.time() < deadline:
                message = consumer.poll(1.0)
                if message is not None and not message.error():
                    break

            assert message is not None
            assert message.value() == b"ready"
        finally:
            consumer.close()
