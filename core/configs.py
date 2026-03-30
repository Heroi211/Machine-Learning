from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
import logging

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
    mlflow_tracking_uri: str = Field(default="sqlite:///artifacts/mlruns/mlflow.db",validation_alias="MLFLOW_TRACKING_URI")
    mlflow_artifact_root: str = Field(default="artifacts/mlruns",validation_alias="MLFLOW_ARTIFACT_ROOT")
    
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
    
    database_url: str | None = None
    
    jwt_secret: str = Field(validation_alias="SECRET", description="Chave secreta JWT")
    algorithm: str = Field(validation_alias="ALGORITHM", description="Algoritmo JWT")
    access_token_expire_minutes: int = Field(validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES", description="Minutos para expiração do token de acesso")
    
    timezone: str = Field(validation_alias="TIMEZONE", description="Timezone para logs e timestamps")
    
    log_http_requests: bool = Field(default=True,validation_alias="LOG_HTTP_REQUESTS",description="Log de método, path, status e latência por requisição")

    log_http_requests_file: bool = Field(
        default=True,
        validation_alias="LOG_HTTP_REQUESTS_FILE",
        description="Persistir linhas JSONL de acesso em path_api_request_logs/access.jsonl",
    )
    path_api_request_logs: str = Field(
        default="logs/api_requests",
        validation_alias="PATH_API_REQUEST_LOGS",
        description="Diretório dos arquivos access.jsonl (rotação automática)",
    )
    log_http_requests_max_bytes: int = Field(
        default=5_242_880,
        validation_alias="LOG_HTTP_REQUESTS_MAX_BYTES",
        description="Tamanho máximo de access.jsonl antes da rotação (~5 MiB)",
    )
    log_http_requests_backup_count: int = Field(
        default=5,
        validation_alias="LOG_HTTP_REQUESTS_BACKUP_COUNT",
        description="Número de arquivos access.jsonl.* retidos após rotação",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_log_level(self) -> int:
        """Retorna o nível de logging baseado em debug""" 
        return logging.DEBUG if self.debug else logging.INFO
    
    @model_validator(mode="after")
    def set_database_url(self):
        self.database_url = (
            f"postgresql+asyncpg://{self.database_user}:"
            f"{self.database_pass}@{self.database_server}:"
            f"{self.database_port}/{self.database_name}"
        )
        return self
    

# Instância global (singleton pattern)
settings:Settings = Settings()
