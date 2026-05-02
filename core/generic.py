"""Define a base compartilhada dos modelos ORM."""

from sqlalchemy import Column, Integer, Boolean, DateTime
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from services.utils import utcnow


@as_declarative()
class modelsGeneric:
    """Fornece campos comuns de auditoria para os modelos ORM."""

    created_at = Column(DateTime, default=utcnow, nullable=False)
    active = Column(Boolean, default=True)

    @declared_attr
    def __tablename__(cls):
        """Infere o nome da tabela a partir do nome da classe em minusculas."""
        return cls.__name__.lower()
