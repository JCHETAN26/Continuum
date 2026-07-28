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


def test_python_and_node_images_run_as_non_root():
    python_dockerfile = Path("docker/Dockerfile.python").read_text()
    node_dockerfile = Path("docker/Dockerfile.node").read_text()

    assert "USER continuum" in python_dockerfile
    assert "USER node" in node_dockerfile
