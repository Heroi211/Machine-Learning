"""
Configuração do logger `api.request`: console (propagação ao root) + arquivo JSON rotacionado.

Uma linha JSON por requisição facilita scripts de percentis (p50, p90, etc.).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os

from core.configs import settings

_CONFIGURED = False


class JsonAccessFormatter(logging.Formatter):
    """Uma linha JSON por registro quando `access_payload` está em `extra`."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize structured access payloads as JSON strings."""
        payload = getattr(record, "access_payload", None)
        if payload is not None:
            return json.dumps(payload, ensure_ascii=False)
        return super().format(record)


def setup_api_request_logging() -> None:
    """Anexa `RotatingFileHandler` ao logger `api.request` (idempotente)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    if not settings.log_http_requests_file:
        _CONFIGURED = True
        return

    log_dir = settings.path_api_request_logs
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "access.jsonl")

    file_handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=settings.log_http_requests_max_bytes,
        backupCount=settings.log_http_requests_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonAccessFormatter())
    file_handler.setLevel(logging.INFO)

    lg = logging.getLogger("api.request")
    lg.addHandler(file_handler)
    lg.setLevel(logging.INFO)
    lg.propagate = True

    _CONFIGURED = True
