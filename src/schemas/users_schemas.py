"""Schemas Pydantic para entrada e saída de dados de usuários."""

from pydantic import BaseModel as SC_BaseModel
from typing import Optional
from pydantic import EmailStr
from datetime import datetime


class users(SC_BaseModel):
    """Schema completo de usuário retornado pela API."""

    id:Optional[int] = None
    name: str
    email:EmailStr
    created_at:Optional[datetime] = datetime.now()
    active:Optional[bool] = True
    role_id: Optional[int] = 1
    class Config:
        from_attributes = True

class users_update(users):
    """Schema de atualização parcial de usuário."""

    name:Optional[str] = None
    email:Optional[EmailStr] = None
    password:Optional[str] = None
    active:Optional[bool] = True
    role_id:Optional[int] = None

class users_create(users):
    """Schema de criação de usuário com senha obrigatória."""

    password:str

class usersGetData(SC_BaseModel):
    """Schema resumido de usuário com papel renderizado."""

    id:Optional[int] = None
    name:Optional[str] = None
    email:Optional[EmailStr] = None
    active:Optional[bool] = True
    role:Optional[str] = None
