"""Local-first HTTP boundary. No accounts, billing, provisioning, or retries."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import os
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import MAX_BODY_BYTES
from .contracts import DjevRequest
from .engine import BackendError, CapacityError, SchemaError


def _error(status: int, message: str, **headers) -> JSONResponse:
    return JSONResponse({"error": {"message": message}}, status_code=status, headers=headers)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object keys must be unique")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("JSON numbers must be finite")


def create_app(*, engine=None, engine_factory=None, api_key: str | None = None,
               static_dir: Path | None = None, max_active_requests: int = 8,
               timeout_seconds: float = 120) -> FastAPI:
    """Inject an engine for tests; normal execution lazily creates the real adapter."""
    if engine_factory is None:
        from .multimodal import create_engine
        engine_factory = create_engine
    if api_key is None:
        api_key = os.environ.get("DJEV_API_KEY", "")
    if type(max_active_requests) is not int or max_active_requests < 1:
        raise ValueError("max_active_requests must be a positive integer")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    engine_lock = asyncio.Lock()
    active = 0

    async def get_engine():
        nonlocal engine
        async with engine_lock:
            if engine is None:
                engine = await engine_factory()
        return engine

    @asynccontextmanager
    async def lifespan(app):
        yield
        if engine is not None:
            await engine.close()

    app = FastAPI(title="Djev", version="0.1.0", lifespan=lifespan,
                  description="Typed Noul, Choice and Score decisions with native images.")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def authorized(request: Request) -> bool:
        if not api_key:
            return True
        values = request.headers.getlist("authorization")
        return len(values) == 1 and hmac.compare_digest(
            values[0].encode(), ("Bearer " + api_key).encode())

    @app.get("/health")
    async def health():
        return {"status": "alive", "service": "djev", "model": "djev-0.1"}

    @app.get("/ready")
    async def ready():
        try:
            async with asyncio.timeout(timeout_seconds):
                available = await (await get_engine()).health()
        except Exception:
            available = False
        return JSONResponse({"ready": available}, status_code=200 if available else 503)

    @app.get("/config")
    async def config():
        return {"api_path": "/v1/request", "model": "djev-0.1", "auth_required": bool(api_key),
                "limits": {"state_characters": 20000, "instructions_characters": 2000,
                           "criterion_characters": 500, "questions": 32, "images": 6,
                           "state_images": 1, "image_bytes": 5 * 1024 * 1024,
                           "image_dimension": 2048, "body_bytes": MAX_BODY_BYTES},
                "features": {"images": True, "question_images": True, "durable_requests": False}}

    @app.post("/v1/request", openapi_extra={"requestBody": {"required": True, "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/DjevRequest"}}}}})
    async def evaluate(request: Request):
        nonlocal active
        if not authorized(request):
            return _error(401, "A valid API key is required", **{"WWW-Authenticate": "Bearer"})
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            return _error(415, "Content-Type must be application/json")
        lengths = request.headers.getlist("content-length")
        try:
            if len(lengths) > 1 or (lengths and (int(lengths[0]) < 0 or int(lengths[0]) > MAX_BODY_BYTES)):
                return _error(413, "Request body exceeds 8 MiB")
        except ValueError:
            return _error(400, "Invalid Content-Length")
        if active >= max_active_requests:
            return _error(503, "Request capacity is full; retry later", **{"Retry-After": "1"})
        # Admission is atomic on the serving event loop and precedes body reads.
        active += 1
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_seconds):
                data = bytearray()
                async for chunk in request.stream():
                    if len(data) + len(chunk) > MAX_BODY_BYTES:
                        return _error(413, "Request body exceeds 8 MiB")
                    data.extend(chunk)
                try:
                    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                    parsed = DjevRequest.model_validate(value)
                except ValidationError as exc:
                    # Never reflect state, image bytes or secrets in error responses.
                    return JSONResponse({"error": {"message": "Invalid decision request", "issues": [
                        {"path": list(item["loc"]), "message": item["msg"]}
                        for item in exc.errors(include_input=False, include_context=False, include_url=False)]}},
                        status_code=422)
                except (ValueError, RecursionError, UnicodeDecodeError):
                    return _error(400, "Body must be valid JSON with unique keys and finite numbers")
                try:
                    current = await get_engine()
                except Exception:
                    return _error(503, "Model adapter is unavailable; check the server runtime configuration")
                result = await current.generate(parsed)
                elapsed = (time.perf_counter() - started) * 1000
                return JSONResponse(result.body, headers={
                    "Server-Timing": f"compile;dur={result.compile_ms:.3f},model;dur={result.model_ms:.3f},total;dur={elapsed:.3f}",
                    "X-Djev-Model-Ms": f"{result.model_ms:.3f}",
                    "X-Djev-Server-Ms": f"{elapsed:.3f}",
                })
        except CapacityError:
            return _error(503, "Inference capacity is full; retry later", **{"Retry-After": "1"})
        except SchemaError as exc:
            return _error(422, str(exc))
        except BackendError:
            return _error(502, "Inference backend did not return complete valid evidence")
        except TimeoutError:
            return _error(504, "Request deadline exceeded; no automatic retry was attempted")
        finally:
            active -= 1

    static_dir = static_dir if static_dir is not None else Path(__file__).resolve().parents[1] / "playground" / "dist"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="playground")

    def openapi():
        if app.openapi_schema is None:
            document = get_openapi(title=app.title, version=app.version,
                                   description=app.description, routes=app.routes)
            schema = DjevRequest.model_json_schema(ref_template="#/components/schemas/{model}")
            definitions = schema.pop("$defs", {})
            definitions["DjevRequest"] = schema
            document.setdefault("components", {}).setdefault("schemas", {}).update(definitions)
            app.openapi_schema = document
        return app.openapi_schema

    app.openapi = openapi
    return app
