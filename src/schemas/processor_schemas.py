from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class PipelineRunResponse(BaseModel):
    id: int
    user_id: int
    pipeline_type: str
    is_airflow_run: bool = False
    objective: str
    status: str
    original_filename: str
    model_path: Optional[str] = None
    csv_output_path: Optional[str] = None
    metrics: Optional[dict] = None
    error_message: Optional[str] = None
    active: bool = Field(default=True, description="Run lógico ativo no painel interno.")
    inference_backend: str = Field(
        default="sklearn",
        description="Backend servido em /predict para este run: 'sklearn' (joblib) ou 'mlp' (PyTorch).",
    )
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
    Para /predict, o corpo usa ``PredictRequest`` (união discriminada por ``domain``).
    """

    heart_disease = "heart_disease"
    churn = "churn"


# Nome legado usado em discussões / checklist — mesmo tipo.
PredictDomain = MLDomain


class ChurnFeaturesInput(BaseModel):
    """
    **Entrada do joblib (passo ``preprocess``):** uma linha ao nível de ``train_features_pre_transform.csv``,
    **mas só as colunas “brutas”** antes das criadas pela strategy (sem ``is_new_customer``, ``tenure_log``, …).

    **Não** use valores desta planilha: ``train_model_input.csv`` (já escalados + one-hot internos do sklearn) —
    isso seria **reaplicar** StandardScaler/OHE e destrói a predição.

    Chaves em minúsculas, alinhadas ao CSV do baseline/Telco.
    """

    model_config = ConfigDict(extra="forbid")

    gender: str = Field(..., description="Ex.: Male, Female")
    seniorcitizen: int = Field(..., ge=0, le=1, description="0 ou 1 (codificação binária)")
    partner: int = Field(..., ge=0, le=1)
    dependents: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    phoneservice: int = Field(..., ge=0, le=1)
    multiplelines: int | str = Field(
        ...,
        description="0/1 se o treino usou binário; caso contrário alinhar ao CSV de treino",
    )
    internetservice: str = Field(..., description="Ex.: DSL, Fiber optic, No")
    onlinesecurity: int = Field(..., ge=0, le=1)
    onlinebackup: int = Field(..., ge=0, le=1)
    deviceprotection: int = Field(..., ge=0, le=1)
    techsupport: int = Field(..., ge=0, le=1)
    streamingtv: int = Field(..., ge=0, le=1)
    streamingmovies: int = Field(..., ge=0, le=1)
    contract: str = Field(..., description="Ex.: Month-to-month, One year, Two year")
    paperlessbilling: int = Field(..., ge=0, le=1)
    paymentmethod: str = Field(..., description="Ex.: Electronic check, Credit card (automatic)")
    monthlycharges: float = Field(..., ge=0)
    totalcharges: float = Field(..., ge=0)


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


class PredictRequestChurn(BaseModel):
    """Pedido de predição — domínio churn (Telecom)."""

    model_config = ConfigDict(extra="forbid")

    domain: Literal["churn"] = Field(..., description="Domínio com modelo FE promovido")
    features: ChurnFeaturesInput = Field(..., description="Atributos brutos alinhados ao treino (Telco)")


class PredictRequestHeartDisease(BaseModel):
    """Pedido de predição — domínio cardiologia (legado)."""

    model_config = ConfigDict(extra="forbid")

    domain: Literal["heart_disease"] = Field(..., description="Domínio heart disease")
    features: HeartDiseaseFeaturesInput = Field(..., description="Atributos (formato one-hot do treino)")


PredictRequest = Annotated[
    Union[PredictRequestChurn, PredictRequestHeartDisease],
    Field(discriminator="domain"),
]


class MetricSnapshot(BaseModel):
    """Métricas de classificação binária no holdout de treino (um único split de teste)."""

    model_config = ConfigDict(extra="ignore")

    accuracy: Optional[float] = Field(default=None, description="Acurácia no conjunto de teste do run.")
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None


class InferenceReport(BaseModel):
    """
    Mini-relatório para o consumidor de ``/predict``: o que está servido e o que foi medido no treino.

    **Nota:** probabilidade e classe deste JSON são da **linha atual**; blocos ``*_holdout*``
    vêm das métricas gravadas no ``PipelineRun`` (histórico do treino).
    """

    model_config = ConfigDict(extra="ignore")

    inference_backend: str = Field(..., description="`sklearn` ou `mlp`.")
    predict_model: str = Field(..., description="Identificador lógico, ex.: sklearn_pipeline ou pytorch_mlp.")
    sklearn_benchmark_classifier: Optional[str] = Field(
        default=None,
        description="Nome do melhor estimador sklearn na fase de CV / estudo (pode não ser o servido).",
    )
    optimization_metric: Optional[str] = None
    best_cv_score: Optional[float] = Field(default=None, description="Melhor média de CV na seleção interna.")
    classification_decision_threshold: Optional[float] = Field(
        default=None,
        description="Limiar usado ao calcular métricas de teste sklearn (holdout), se registado.",
    )
    served_holdout_metrics: Optional[MetricSnapshot] = Field(
        default=None,
        description="Métricas de teste do **modelo efectivamente servido** (sklearn após tuning ou MLP).",
    )
    sklearn_holdout_test: Optional[MetricSnapshot] = Field(
        default=None,
        description="Quando o servido é MLP: métricas holdout do sklearn no mesmo run (se existirem).",
    )
    baseline_reference: Optional[dict[str, Any]] = Field(
        default=None,
        description="Snapshot do PipelineRun **baseline** activo no momento do treino FE (metadados + test_*).",
    )
    baseline_holdout_metrics: Optional[MetricSnapshot] = Field(
        default=None,
        description="Métricas de teste do Baseline (sklearn simples), para comparar com o modelo servido.",
    )
    fe_model_comparison: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tabela do FE: modelos sklearn pré/pós-tuning e MLP (holdout de teste do run).",
    )
    mlp_training_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="Resumo do treino MLP (épocas, val_loss, hiperparâmetros) quando aplicável.",
    )
    notes: list[str] = Field(default_factory=list, description="Avisos e contexto para interpretação.")
    summary_lines: list[str] = Field(
        default_factory=list,
        description="Frases curtas para UI ou relatório (Markdown leve).",
    )


class PredictResponse(BaseModel):
    id: int
    domain: str
    pipeline_run_id: int
    prediction: int
    probability: Optional[float] = Field(
        default=None,
        description="Probabilidade estimada da classe positiva em percentual (0–100), quando disponível.",
    )
    input_data: dict
    inference_report: InferenceReport = Field(
        ...,
        description="Contexto do modelo servido e métricas de holdout do treino associadas a este run.",
    )

    class Config:
        from_attributes = True


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


class TriggerDagRequest(BaseModel):
    objective: MLDomain = Field(description="Domínio do problema (mesmos valores do treino / Airflow).")
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Field(
        default="accuracy",
        description="Métrica para seleção do melhor modelo no FE.",
    )
    min_precision: Optional[float] = Field(default=None, description="Guardrail opcional: precisão mínima [0,1].")
    min_roc_auc: Optional[float] = Field(default=None, description="Guardrail opcional: ROC-AUC mínimo [0,1].")
    tuning_n_iter: Optional[int] = Field(default=None, description="Número máximo de amostras no tuning (opcional).")
    time_limit_minutes: int = Field(default=2, description="Orçamento de tempo para tuning (minutos).")
    acc_target: Optional[float] = Field(default=None, description="Alvo opcional da métrica primária no tuning.")


class TriggerDagResponse(BaseModel):
    dag_run_id: str
    dag_id: str
    objective: str
    csv_path: str
    message: str
