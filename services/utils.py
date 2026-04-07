import pytz
from datetime import datetime, time
from typing import Optional


def to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None

    if dt.tzinfo is None:
        sp = pytz.timezone("America/Sao_Paulo")
        dt = sp.localize(dt)
    dt_utc = dt.astimezone(pytz.UTC).replace(tzinfo=None)
    return dt_utc

def utcnow() -> datetime:
    return datetime.now(pytz.UTC).replace(tzinfo=None)

    
