from pydantic_settings import BaseSettings
from pydantic import Field
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
    
    debug: bool = Field(default=False,validation_alias="DEBUG",description="Ativa modo debug"    )
    test_size: float = Field(default=0.2,validation_alias="TEST_SIZE",description="Proporção de teste (0.0 a 1.0)")
    random_state: int = Field(default=42,validation_alias="RANDOM_STATE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def get_log_level(self) -> int:
        """Retorna o nível de logging baseado em debug"""
        return logging.DEBUG if self.debug else logging.INFO

# Instância global (singleton pattern)
settings = Settings()