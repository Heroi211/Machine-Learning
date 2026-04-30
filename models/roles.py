"""Define role constants and the roles ORM model."""

from sqlalchemy import Column,Integer,String
from core.generic import modelsGeneric
from sqlalchemy.orm import relationship

class Roles(modelsGeneric):
    """Represent an authorization role assigned to users."""

    USER = 1
    ADMINISTRATOR = 2


    ROLES = [
    (USER, 'Usuario'), # Usuário padrão, sem permissões especiais, mas que atua nas rotinas
    (ADMINISTRATOR, 'Administrador'), # Administrador do sistema com poder total, incluindo gerenciamento de usuários e configurações.
]
    __tablename__='roles'
    id = Column(Integer,primary_key=True,autoincrement=True)
    description = Column(String,nullable=False)
   
    user = relationship("Users",uselist=False, back_populates="role")
    
    def __init__(self,description):
        """Initialize a role with a display description."""
        self.description = description
    
    def get_role_display(self) -> str:
        """Return the human-readable label for the role ID."""
        for code, label in self.ROLES:
            if code == self.id:
                return label
        return "Role inválida"
