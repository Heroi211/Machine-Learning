"""Schemas Pydantic para criação e atualização de papéis."""

from pydantic import BaseModel as SC_basemodel
from typing import Optional


class role(SC_basemodel):
    """Schema de leitura e criação de papéis."""

    id:Optional[int] = None
    description:str
    active:bool

    class Config:
        from_attributes = True

class role_update(role):
    """Schema de atualização parcial de papéis."""

    description:Optional[str] = None
    active:Optional[bool] = None
