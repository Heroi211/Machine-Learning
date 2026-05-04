"""Classe base declarativa com campos comuns para os modelos ORM."""

from sqlalchemy import Column,Boolean,DateTime
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from services.utils import utcnow

@as_declarative()
class modelsGeneric:
    """Base ORM com campos de auditoria compartilhados pelos modelos."""

    created_at = Column(DateTime,default=utcnow,nullable=False)
    active = Column(Boolean,default=True)

    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
