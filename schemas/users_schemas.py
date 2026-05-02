"""Define schemas Pydantic para payloads da API de usuarios."""

from pydantic import BaseModel as SC_BaseModel
from typing import Optional
from pydantic import EmailStr
from datetime import datetime


class users(SC_BaseModel):
    """Representa uma resposta de usuario com identificador de papel."""

    id: Optional[int] = None
    name: str
    email: EmailStr
    created_at: Optional[datetime] = datetime.now()
    active: Optional[bool] = True
    role_id: Optional[int] = 1

    class Config:
        from_attributes = True


class users_update(users):
    """Representa os campos opcionais aceitos na atualizacao de um usuario."""

    name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    active: Optional[bool] = True
    role_id: Optional[int] = None


class users_create(users):
    """Representa os campos obrigatorios para criar um usuario."""

    password: str


class usersGetData(SC_BaseModel):
    """Representa dados de usuario retornados nas rotas de listagem e detalhe."""

    id: Optional[int] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    active: Optional[bool] = True
    role: Optional[str] = None
