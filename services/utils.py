"""Fornece utilitarios compartilhados para datas e horarios."""

import pytz
from datetime import datetime, time
from typing import Optional


def to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Converte um datetime para UTC sem timezone, assumindo São Paulo quando naive."""
    if dt is None:
        return None

    if dt.tzinfo is None:
        sp = pytz.timezone("America/Sao_Paulo")
        dt = sp.localize(dt)
    dt_utc = dt.astimezone(pytz.UTC).replace(tzinfo=None)
    return dt_utc


def utcnow() -> datetime:
    """Retorna o horário atual como datetime UTC sem timezone."""
    return datetime.now(pytz.UTC).replace(tzinfo=None)
