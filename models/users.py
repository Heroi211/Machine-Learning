"""Define o modelo ORM de usuarios."""

from sqlalchemy import Column, Integer, String, Boolean, Date, ForeignKey, Null, DateTime
from sqlalchemy.orm import relationship
from core.generic import modelsGeneric


class Users(modelsGeneric):
    """Representa um usuario autenticado da aplicacao."""

    __tablename__ = 'users'
    id = Column(Integer, autoincrement=True, primary_key=True)
    password = Column(String(255), nullable=False)
    name = Column(String(50), nullable=False)
    email = Column(String(50), nullable=False)
    role_id = Column(Integer, ForeignKey('roles.id'), default=1)

    # Relação da chave estrangeira que associa o usuário ao seu papel.
    role = relationship("Roles", lazy='joined', back_populates='user')
