"""Entrypoint da API na raiz do repo; adiciona ``src/`` ao ``sys.path``."""
# ruff: noqa: E402
from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastapi import FastAPI
from core.configs import settings
from core.logging_setup import setup_root_logging
from core.logging_api_request import setup_api_request_logging
from api.v1 import api
from core.middleware.request_record import request_record

setup_root_logging()
setup_api_request_logging()

app = FastAPI(title=settings.project_name,version=settings.project_version)
app.middleware("http")(request_record)

app.include_router(api.router,prefix=settings.project_version)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)
