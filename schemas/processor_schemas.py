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
    pipeline_run_id: int = Field(description="ID do pipeline_run cujo modelo será usado")
    features: dict = Field(description="Campos do indivíduo. Ex: {'age': 55, 'chol': 250, ...}")


class PredictResponse(BaseModel):
    id: int
    pipeline_run_id: int
    prediction: int
    probability: Optional[float] = None
    input_data: dict

    class Config:
        from_attributes = True
