"""Define Pydantic schemas for user API payloads."""

from pydantic import BaseModel as SC_BaseModel
from typing import Optional
from pydantic import EmailStr
from datetime import datetime


class users(SC_BaseModel):
    """Represent a user response with role identifier."""

    id:Optional[int] = None
    name: str
    email:EmailStr
    created_at:Optional[datetime] = datetime.now()
    active:Optional[bool] = True
    role_id: Optional[int] = 1
    class Config:
        from_attributes = True

class users_update(users):
    """Represent optional fields accepted when updating a user."""

    name:Optional[str] = None
    email:Optional[EmailStr] = None
    password:Optional[str] = None
    active:Optional[bool] = True
    role_id:Optional[int] = None

class users_create(users):
    """Represent fields required to create a user."""

    password:str

class usersGetData(SC_BaseModel):
    """Represent user data returned by list and detail endpoints."""

    id:Optional[int] = None
    name:Optional[str] = None
    email:Optional[EmailStr] = None
    active:Optional[bool] = True
    role:Optional[str] = None
