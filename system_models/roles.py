from sqlalchemy import Column,Integer,String
from core.generic import modelsGeneric
from sqlalchemy.orm import relationship

class Roles(modelsGeneric):
    USER = 1
    OPERATOR = 2
    ADMINISTRATOR = 3
    USER_CLIENT = 4

    ROLES = [
    (USER, 'Usuario'), # Usuário padrão, sem permissões especiais, mas que atua nas rotinas
    (OPERATOR, 'Operador'), # Usuário com permissões para verificar o sistema e aprovar tarefas. 
    (ADMINISTRATOR, 'Administrador'), # Administrador do sistema com poder total
    (USER_CLIENT, 'Usuario_cliente'), # Usuário do sistema, que também é cliente, pode cadastrar rotinas para si e para outros clientes
]
    __tablename__='roles'
    id = Column(Integer,primary_key=True,autoincrement=True)
    description = Column(String,nullable=False)
   
    user = relationship("Users",uselist=False, back_populates="role")
    
    def __init__(self,description):
        self.description = description
    
    def get_role_display(self) -> str:
        for code, label in self.ROLES:
            if code == self.id:
                return label
        return "Role inválida"