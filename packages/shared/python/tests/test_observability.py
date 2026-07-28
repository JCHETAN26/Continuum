from continuum_shared.observability import current_correlation_id, instrument_fastapi
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_correlation_id_is_bound_and_returned() -> None:
    app = FastAPI()
    seen_ids: list[str | None] = []
    instrument_fastapi(app, "test-service")

    @app.get("/ping")
    async def ping() -> dict[str, str | None]:
        seen_ids.append(current_correlation_id())
        return {"correlation_id": current_correlation_id()}

    response = TestClient(app).get("/ping", headers={"x-correlation-id": "request-123"})

    assert response.status_code == 200
    assert response.headers["x-correlation-id"] == "request-123"
    assert response.json() == {"correlation_id": "request-123"}
    assert seen_ids == ["request-123"]
