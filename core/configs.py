from pydantic_settings import BaseSettings
from pydantic import Field
import logging
from sqlalchemy.ext.declarative import declarative_base

class Settings(BaseSettings):
    """
    Configurações centralizadas com validação de tipo automática.
    Carrega do .env e converte para tipos corretos.
    
    Mapeia explicitamente cada field para sua variável de ambiente.
    """
    
    # Caminhos
    path_data: str = Field(default="data/",validation_alias="PATH_DATA",description="Path para dados brutos")
    path_data_preprocessed: str = Field(default="data/pre_processed/",validation_alias="PATH_DATA_PREPROCESSED")
    path_model: str = Field(default="models/",validation_alias="PATH_MODEL")
    path_graphs: str = Field(default="graphs/",validation_alias="PATH_GRAPHS")
    path_logs: str = Field(default="logs/",validation_alias="PATH_LOGS")
    
    debug: bool = Field(default=False,validation_alias="DEBUG",description="Ativa modo debug"    )
    test_size: float = Field(default=0.2,validation_alias="TEST_SIZE",description="Proporção de teste (0.0 a 1.0)")
    random_state: int = Field(default=42,validation_alias="RANDOM_STATE")
    
    project_name: str = Field(validation_alias="PROJECT_NAME", description="Nome do projeto")
    project_version: str = Field(validation_alias="PROJECT_VERSION", description="Versão do projeto")
    
    database_user: str = Field(validation_alias="DATABASE_USER", description="Usuário do banco de dados")
    database_pass: str = Field(validation_alias="DATABASE_PASS", description="Senha do banco de dados")
    database_server: str = Field(validation_alias="DATABASE_SERVER", description="Servidor do banco de dados")
    database_port: int = Field(validation_alias="DATABASE_PORT", description="Porta do banco de dados")
    database_name: str = Field(validation_alias="DATABASE_NAME", description="Nome do banco de dados")
    
    database_url: str = f"postgresql+asyncpg://{database_user}:{database_pass}@{database_server}:{database_port}/{database_name}"
    db_base_model = declarative_base()
    
    jwt_secret: str = Field(validation_alias="SECRET", description="Chave secreta JWT")
    algorithm: str = Field(validation_alias="ALGORITHM", description="Algoritmo JWT")
    access_token_expire_minutes: int = Field(validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES", description="Minutos para expiração do token de acesso")
    
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_log_level(self) -> int:
        """Retorna o nível de logging baseado em debug"""
        return logging.DEBUG if self.debug else logging.INFO

# Instância global (singleton pattern)
settings:Settings = Settings()