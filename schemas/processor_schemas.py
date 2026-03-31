from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


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


class PredictRequest(BaseModel):
    domain: str = Field(
        description="Domínio do modelo em produção (ex.: heart_disease). Deve coincidir com o objective do run promovido."
    )
    features: dict = Field(description="Campos do indivíduo. Ex: {'age': 55, 'chol': 250, ...}")


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
    domain: str = Field(description="Domínio canónico (deve coincidir com o objective do pipeline_run)")
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
