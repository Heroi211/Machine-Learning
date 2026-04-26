from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PipelineRunResponse(BaseModel):
    id: int
    user_id: int
    pipeline_type: str
    objective: str
    status: str
    original_filename: str
    model_path: Optional[str] = None
    csv_output_path: Optional[str] = None
    metrics: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Domínios ML — enum único para OpenAPI (dropdown no Swagger / clientes)
# Alinhar novos valores a STRATEGY_REGISTRY / estratégias de FE antes de treinar.
# ---------------------------------------------------------------------------


class MLDomain(str, Enum):
    """
    Domínios de problema que o produto expõe para treino e operações.
    Ao adicionar um valor: registar strategy em STRATEGY_REGISTRY (FE) e,
    para predição, estender PredictRequest com Union discriminada ou schema por domínio.
    """

    heart_disease = "heart_disease"
    churn = "churn"


# Nome legado usado em discussões / checklist — mesmo tipo.
PredictDomain = MLDomain


class HeartDiseaseFeaturesInput(BaseModel):
    """
    Uma linha no mesmo formato do CSV pós-processado usado no treino (tipos explícitos).
    Chaves com espaço/hífen usam alias para coincidir com o JSON.
    """

    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., description="Idade em anos")
    trestbps: float = Field(..., description="Pressão em repouso (mmHg)")
    chol: float = Field(..., description="Colesterol sérico (mg/dl)")
    fbs: bool = Field(..., description="Açúcar em jejum > 120 mg/dl")
    thalch: float = Field(..., description="Frequência cardíaca máxima alcançada")
    exang: bool = Field(..., description="Angina induzida por exercício")
    oldpeak: float = Field(..., description="Depressão ST induzida por exercício")
    ca: float = Field(..., description="Número de vasos principais coloridos por fluoroscopia")

    sex_Male: bool = Field(..., description="Sexo masculino (one-hot)")

    cp_atypical_angina: bool = Field(..., alias="cp_atypical angina", description="Tipo de dor: angina atípica")
    cp_non_anginal: bool = Field(..., alias="cp_non-anginal", description="Tipo de dor: não anginal")
    cp_typical_angina: bool = Field(..., alias="cp_typical angina", description="Tipo de dor: angina típica")

    restecg_normal: bool = Field(..., description="ECG em repouso: normal")
    restecg_st_t_abnormality: bool = Field(
        ...,
        alias="restecg_st-t abnormality",
        description="ECG em repouso: anormalidade ST-T",
    )

    slope_flat: bool = Field(..., description="Inclinação do segmento ST: flat")
    slope_upsloping: bool = Field(..., description="Inclinação do segmento ST: upsloping")

    thal_normal: bool = Field(..., description="thal: normal")
    thal_reversable_defect: bool = Field(
        ...,
        alias="thal_reversable defect",
        description="thal: defeito reversível",
    )


class PredictRequest(BaseModel):
    """Body válido para predição; expandir com Union discriminada quando houver mais domínios."""

    model_config = ConfigDict(extra="forbid")

    domain: MLDomain = Field(..., description="Domínio com modelo promovido em produção")
    features: HeartDiseaseFeaturesInput = Field(..., description="Atributos do indivíduo (formato do treino)")


class PredictResponse(BaseModel):
    id: int
    domain: str
    pipeline_run_id: int
    prediction: int
    probability: Optional[float] = None
    input_data: dict

    class Config:
        from_attributes = True


class PromoteRequest(BaseModel):
    domain: MLDomain = Field(description="Domínio canónico (deve coincidir com o objective do pipeline_run)")
    pipeline_run_id: int = Field(description="ID do PipelineRuns concluído a promover")


class DeployedModelResponse(BaseModel):
    id: int
    domain: str
    pipeline_run_id: int
    status: str
    promoted_at: Optional[datetime] = None
    promoted_by_user_id: Optional[int] = None
    metrics_snapshot: Optional[dict] = None

    class Config:
        from_attributes = True


class RollbackRequest(BaseModel):
    domain: MLDomain = Field(description="Domínio a reverter para o deployment anterior (archived mais recente).")


class TriggerDagRequest(BaseModel):
    objective: MLDomain = Field(description="Domínio do problema (mesmos valores do treino / Airflow).")
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Field(
        default="accuracy",
        description="Métrica para seleção do melhor modelo no FE.",
    )
    min_precision: Optional[float] = Field(default=None, description="Guardrail opcional: precisão mínima [0,1].")
    min_roc_auc: Optional[float] = Field(default=None, description="Guardrail opcional: ROC-AUC mínimo [0,1].")
    time_limit_minutes: int = Field(default=2, description="Orçamento de tempo para tuning (minutos).")
    acc_target: Optional[float] = Field(default=None, description="Alvo opcional da métrica primária no tuning.")


class TriggerDagResponse(BaseModel):
    dag_run_id: str
    dag_id: str
    objective: str
    csv_path: str
    message: str
