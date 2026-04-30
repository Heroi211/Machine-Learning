"""Define the shared SQLAlchemy base mixin for ORM models."""

from sqlalchemy import Column,Integer, Boolean,DateTime
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from services.utils import utcnow

@as_declarative()
class modelsGeneric:
    """Provide common audit columns for ORM models."""

    created_at = Column(DateTime,default=utcnow,nullable=False)
    active = Column(Boolean,default=True)
    
    @declared_attr
    def __tablename__(cls):
        """Infer table names from lowercase class names."""
        return cls.__name__.lower()
