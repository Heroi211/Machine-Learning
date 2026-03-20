from pydantic import BaseModel as SC_BaseModel
from typing import Optional
from pydantic import EmailStr
from datetime import datetime


class users(SC_BaseModel):
    id:Optional[int] = None
    name: str
    email:EmailStr
    cpf: str
    phone: str
    created_at:Optional[datetime] = datetime.now()
    active:Optional[bool] = True
    role_id: Optional[int] = 1
    reset_password_token:Optional[str] = None
    reset_password_expires:Optional[datetime] = None
    class Config:
        from_attributes = True


class users_update(users):
    name:Optional[str] = None
    email:Optional[EmailStr] = None
    cpf: Optional[str] = None
    password:Optional[str] = None
    phone:Optional[str] = None
    active:Optional[bool] = True
    role_id:Optional[int] = None
    reset_password_token:Optional[str] = None
    reset_password_expires:Optional[datetime] = None

class users_create(users):
    password:str
    phone:str

class usersGetData(users):
    name:Optional[str] = None
    email:Optional[EmailStr] = None
    cpf: Optional[str] = None
    phone:Optional[str] = None
    active:Optional[bool] = True
    role:Optional[str] = None
    tarefas:Optional[int] = None

class users_updateForm(SC_BaseModel):
    name:Optional[str] = None
    email:Optional[EmailStr] = None
    cpf: Optional[str] = None
    phone:Optional[str] = None
    active:Optional[bool] = True
    class Config:
        from_attributes = True