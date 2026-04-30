"""Define Pydantic schemas for role API payloads."""

from pydantic import BaseModel as SC_basemodel
from typing import Optional


class role(SC_basemodel):
    """Represent a role payload or response."""

    id:Optional[int] = None
    description:str
    active:bool
    
    class Config:
        from_attributes = True
        
class role_update(role):
    """Represent fields accepted when updating a role."""

    description:Optional[str] = None
    active:Optional[bool] = None
