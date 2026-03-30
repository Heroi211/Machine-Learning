"""Middleware HTTP: latência e metadados mínimos (sem corpo nem Authorization)."""

import logging
import time
from datetime import datetime, timezone

from starlette.requests import Request

from core.configs import settings

logger = logging.getLogger("api.request")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def request_record(request: Request, call_next):
    if not settings.log_http_requests:
        return await call_next(request)

    start = time.perf_counter()
    client = request.client.host if request.client else "-"

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        payload = {
            "ts": _utc_iso(),
            "method": request.method,
            "path": request.url.path,
            "status": None,
            "client": client,
            "duration_ms": round(elapsed_ms, 3),
            "error": True,
        }
        logger.exception(
            "request failed | %s %s | client=%s | %.2fms",
            request.method,
            request.url.path,
            client,
            elapsed_ms,
            extra={"access_payload": payload},
        )
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000
    payload = {
        "ts": _utc_iso(),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "client": client,
        "duration_ms": round(elapsed_ms, 3),
        "error": False,
    }
    logger.info(
        "%s %s | status=%s | client=%s | %.2fms",
        request.method,
        request.url.path,
        response.status_code,
        client,
        elapsed_ms,
        extra={"access_payload": payload},
    )
    return response