from pathlib import Path

import yaml


def load_compose() -> dict:
    return yaml.safe_load(Path("infra/docker-compose.yml").read_text())


def test_app_services_have_healthchecks():
    services = load_compose()["services"]

    for service_name in [
        "ingest",
        "ingest-worker",
        "embedding",
        "drift",
        "linguistic-drift",
        "trainer-api",
        "trainer-worker",
        "server",
        "retention-worker",
        "dashboard",
    ]:
        healthcheck = services[service_name].get("healthcheck")
        assert healthcheck is not None, f"{service_name} is missing a healthcheck"
        assert healthcheck["retries"] >= 10


def test_app_services_have_resource_limits():
    services = load_compose()["services"]

    for service_name in [
        "ingest",
        "ingest-worker",
        "embedding",
        "drift",
        "linguistic-drift",
        "trainer-api",
        "trainer-worker",
        "server",
        "retention-worker",
        "dashboard",
    ]:
        service = services[service_name]
        assert service.get("cpus"), f"{service_name} is missing a CPU limit"
        assert service.get("mem_limit"), f"{service_name} is missing a memory limit"


def test_dashboard_waits_for_api_health():
    dashboard = load_compose()["services"]["dashboard"]

    assert dashboard["depends_on"]["drift"]["condition"] == "service_healthy"
    assert dashboard["depends_on"]["linguistic-drift"]["condition"] == "service_healthy"
    assert dashboard["depends_on"]["trainer-api"]["condition"] == "service_healthy"


def test_app_services_wait_for_migrations():
    services = load_compose()["services"]

    assert services["migrations"]["command"] == [
        "uv",
        "run",
        "alembic",
        "-c",
        "packages/shared/alembic.ini",
        "upgrade",
        "head",
    ]
    for service_name in [
        "ingest",
        "ingest-worker",
        "embedding",
        "drift",
        "linguistic-drift",
        "trainer-api",
        "trainer-worker",
        "server",
        "retention-worker",
    ]:
        assert (
            services[service_name]["depends_on"]["migrations"]["condition"]
            == "service_completed_successfully"
        )


LONG_RUNNING_SERVICES = [
    "redpanda",
    "redpanda-console",
    "postgres",
    "redis",
    "minio",
    "ingest",
    "ingest-worker",
    "embedding",
    "drift",
    "linguistic-drift",
    "trainer-api",
    "trainer-worker",
    "server",
    "retention-worker",
    "dashboard",
]

# These run to completion and are awaited via service_completed_successfully. A restart
# policy would put them in a loop and hold the dependent services back forever.
ONE_SHOT_SERVICES = ["redpanda-init", "minio-init", "migrations"]


def test_long_running_services_restart_on_failure():
    services = load_compose()["services"]

    for service_name in LONG_RUNNING_SERVICES:
        assert services[service_name].get("restart") == "unless-stopped", (
            f"{service_name} would stay down after a crash"
        )


def test_one_shot_services_have_no_restart_policy():
    services = load_compose()["services"]

    for service_name in ONE_SHOT_SERVICES:
        assert "restart" not in services[service_name], (
            f"{service_name} runs once; a restart policy would loop it"
        )


def test_database_url_pins_connection_pool_size():
    compose = load_compose()
    database_url = compose["x-app-env"]["DATABASE_URL"]

    # Left implicit, Prisma derives the pool from visible CPUs, which the per-service
    # cpus limits shrink to 3 and the drift services then exhaust.
    assert "connection_limit=10" in database_url
    assert "pool_timeout=30" in database_url


def test_postgres_allows_headroom_over_the_summed_pools():
    postgres = load_compose()["services"]["postgres"]

    assert postgres["command"] == ["postgres", "-c", "max_connections=200"]


def test_trigger_threshold_is_reachable_by_scores_that_raise_an_alert():
    """The throttler gate and the alert threshold must share a scale.

    Both are centroid cosine distance. If the throttler demands more drift than the
    drift service alerts on, every alert is suppressed and only linguistic drift can
    ever trigger training.
    """
    app_env = load_compose()["x-app-env"]

    alert_threshold = float(app_env["DRIFT_THRESHOLD"])
    trigger_threshold = float(app_env["DRIFT_TRIGGER_MIN_EMBEDDING_DRIFT"])

    assert trigger_threshold <= alert_threshold, (
        f"drift alerts fire at {alert_threshold} but training needs {trigger_threshold}, "
        "so embedding drift can never trigger a training job"
    )


def test_python_and_node_images_run_as_non_root():
    python_dockerfile = Path("docker/Dockerfile.python").read_text()
    node_dockerfile = Path("docker/Dockerfile.node").read_text()

    assert "USER continuum" in python_dockerfile
    assert "USER node" in node_dockerfile


SCALABLE_SERVICES = ["embedding", "ingest-worker"]

# Scaling these would be wrong rather than merely useless: the one-shots must run exactly
# once for service_completed_successfully to mean anything, and a second retention worker
# would duplicate the deletions the first is already making.
SINGLETON_SERVICES = ["redpanda-init", "minio-init", "migrations", "retention-worker"]


def test_scalable_workers_have_no_fixed_container_name():
    """docker compose refuses to scale a service that declares container_name."""
    services = load_compose()["services"]

    for service_name in SCALABLE_SERVICES:
        assert "container_name" not in services[service_name], (
            f"{service_name} cannot be scaled while it declares a container_name"
        )


def test_scalable_workers_publish_no_ports():
    """A published host port collides on the second replica."""
    services = load_compose()["services"]

    for service_name in SCALABLE_SERVICES:
        assert not services[service_name].get("ports"), (
            f"{service_name} cannot be scaled while it publishes a host port"
        )


def test_singleton_services_keep_their_fixed_names():
    """Naming them is what stops a careless --scale from running them more than once."""
    services = load_compose()["services"]

    for service_name in SINGLETON_SERVICES:
        assert services[service_name].get("container_name"), f"{service_name} must not be scalable"


def test_every_prisma_migration_is_mounted_into_postgres():
    """Compose applies the product schema through init scripts, not a migrate step.

    A migration that exists on disk but is not mounted simply never runs, and the failure
    surfaces far away: services start, then fail at query time with a column that does not
    exist. That is how first_embedded_at reached CI.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    migrations = sorted(
        path.name
        for path in (root / "packages/shared/prisma/migrations").iterdir()
        if path.is_dir()
    )

    mounts = " ".join(str(volume) for volume in load_compose()["services"]["postgres"]["volumes"])

    for migration in migrations:
        assert migration in mounts, (
            f"{migration}/migration.sql is not mounted into postgres, so it will never run"
        )


def test_mounted_migrations_run_in_directory_order():
    """Prisma names migrations by timestamp; the init scripts must apply them in that order."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    migrations = sorted(
        path.name
        for path in (root / "packages/shared/prisma/migrations").iterdir()
        if path.is_dir()
    )

    volumes = [
        str(volume)
        for volume in load_compose()["services"]["postgres"]["volumes"]
        if "initdb" in str(volume) and "migration.sql" in str(volume)
    ]
    # The numeric prefix on the init script decides execution order.
    prefixes = [volume.split("/docker-entrypoint-initdb.d/")[1][:2] for volume in volumes]

    assert prefixes == sorted(prefixes)
    assert len(volumes) == len(migrations)


def test_measurement_steps_do_not_hide_failures_behind_tee():
    """`cmd | tee file` exits with tee's status, so a crashed benchmark reports success.

    The load test did exactly that: it died on a check-constraint violation and the job
    went green in 29 seconds. The retrieval benchmark had been doing the same thing
    quietly, which is why one run produced neither output nor its results artifact while
    its step showed as passed.
    """
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    piped = [line for line in workflow.splitlines() if "| tee " in line]
    assert piped, "expected at least one measurement step piping into tee"
    assert workflow.count("set -o pipefail") >= len(piped)
