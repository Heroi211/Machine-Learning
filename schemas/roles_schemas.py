"""Define schemas Pydantic para payloads da API de papeis."""

from pydantic import BaseModel as SC_basemodel
from typing import Optional


class role(SC_basemodel):
    """Representa o payload ou a resposta de um papel."""

    id: Optional[int] = None
    description: str
    active: bool

    class Config:
        from_attributes = True


class role_update(role):
    """Representa os campos aceitos na atualizacao de um papel."""

    description: Optional[str] = None
    active: Optional[bool] = None
