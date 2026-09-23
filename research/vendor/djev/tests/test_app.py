import asyncio
import json

import httpx
import pytest

from djev.app import create_app
from djev.engine import BackendError, EngineResult

BODY = {"state": "Please refund the duplicate payment.",
        "questions": {"refund": {"type": "noul", "instructions": "Requests a refund?"}}}


class Engine:
    """Boundary-only test double: no model quality or GPU claims."""
    def __init__(self):
        self.calls = []

    async def generate(self, request):
        self.calls.append(request)
        return EngineResult({"model": "djev-0.1", "answers": {"refund": {"type": "noul", "noul": .75}},
                             "usage": {"input_tokens": 20, "output_tokens": 8}}, 1, 2)

    async def health(self):
        return True

    async def close(self):
        pass


async def test_request_success_default_seed_health_and_honest_timing():
    engine = Engine()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(engine=engine)), base_url="http://test") as client:
        assert (await client.get("/health")).json()["status"] == "alive"
        assert (await client.get("/ready")).json() == {"ready": True}
        result = await client.post("/v1/request", json=BODY)
        assert result.status_code == 200
        assert result.json()["answers"]["refund"]["noul"] == .75
        assert result.headers["cache-control"] == "no-store"
        assert result.headers["x-content-type-options"] == "nosniff"
        assert "model;dur=2.000" in result.headers["server-timing"]
        assert engine.calls[0].options.seed == 0
        assert (await client.get("/v1/machine")).status_code == 404


@pytest.mark.parametrize("body,status", [
    ('{"state":"one","state":"two","questions":{}}', 400),
    ('{"state":[NaN],"questions":{}}', 400),
    ('{', 400),
    (json.dumps({**BODY, "private_control": "hidden-value"}), 422),
    (json.dumps({**BODY, "model": "unknown-model"}), 422),
])
async def test_invalid_body_never_reaches_engine_or_echoes_payload(body, status):
    engine = Engine()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(engine=engine)), base_url="http://test") as client:
        result = await client.post("/v1/request", content=body, headers={"Content-Type": "application/json"})
        assert result.status_code == status
        assert "hidden-value" not in result.text
        assert not engine.calls


async def test_optional_bearer_key_precedes_body_parsing_and_rejects_duplicates():
    engine = Engine()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(engine=engine, api_key="test-only-placeholder")), base_url="http://test") as client:
        assert (await client.post("/v1/request", content="{")).status_code == 401
        headers = [("Authorization", "Bearer test-only-placeholder")] * 2
        assert (await client.post("/v1/request", json=BODY, headers=headers)).status_code == 401
        assert (await client.post("/v1/request", json=BODY, headers={"Authorization": "Bearer test-only-placeholder"})).status_code == 200
        assert len(engine.calls) == 1


async def test_capacity_rejects_without_waiting_and_releases_after_cancelled_deadline():
    entered = asyncio.Event()
    class Waiting(Engine):
        async def generate(self, request):
            entered.set()
            await asyncio.Event().wait()
    app = create_app(engine=Waiting(), max_active_requests=1, timeout_seconds=.1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        first = asyncio.create_task(client.post("/v1/request", json=BODY))
        await entered.wait()
        rejected = await client.post("/v1/request", json=BODY)
        assert rejected.status_code == 503 and rejected.headers["retry-after"] == "1"
        assert (await first).status_code == 504
        assert (await client.post("/v1/request", json=BODY)).status_code == 504


async def test_untrusted_backend_error_is_not_reflected():
    class Broken(Engine):
        async def generate(self, request):
            raise BackendError("upstream secret or request content")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(engine=Broken())), base_url="http://test") as client:
        result = await client.post("/v1/request", json=BODY)
        assert result.status_code == 502
        assert "secret" not in result.text


async def test_streaming_body_limit_and_content_type_apply_before_inference(monkeypatch):
    import djev.app
    monkeypatch.setattr(djev.app, "MAX_BODY_BYTES", 32)
    engine = Engine()
    async def chunks():
        yield b" " * 20
        yield b" " * 20
    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(engine=engine)), base_url="http://test") as client:
        assert (await client.post("/v1/request", json=BODY)).status_code == 413
        assert (await client.post("/v1/request", content=chunks(), headers={"Content-Type": "application/json"})).status_code == 413
        assert (await client.post("/v1/request", content="{}")).status_code == 415
        assert not engine.calls


async def test_openapi_references_resolve_and_static_mount_is_explicit(tmp_path):
    (tmp_path / "index.html").write_text("<h1>Local playground</h1>")
    app = create_app(engine=Engine(), static_dir=tmp_path)
    document = app.openapi()
    def walk(value):
        if isinstance(value, dict):
            if "$ref" in value:
                target = document
                for part in value["$ref"].removeprefix("#/").split("/"):
                    target = target[part]
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(document)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
        assert "Local playground" in (await client.get("/")).text
        assert (await client.get("/health")).status_code == 200
