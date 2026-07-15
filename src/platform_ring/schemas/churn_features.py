"""Payload de predict — domínio churn."""

from pydantic import BaseModel, ConfigDict, Field


class ChurnFeaturesInput(BaseModel):
    """
    Entrada do joblib (passo ``preprocess``): uma linha ao nível de ``train_features_pre_transform.csv``,
    mas só as colunas brutas antes das criadas pela strategy.

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
