"""Modelo ORM para execuções de pipelines de treino e processamento."""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from core.generic import modelsGeneric


class PipelineRuns(modelsGeneric):
    """Representa uma execução de pipeline registrada no banco."""

    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pipeline_type = Column(String(50), nullable=False)
    objective = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False, default="processing")
    original_filename = Column(String(255), nullable=False)
    model_path = Column(String(500), nullable=True)
    csv_output_path = Column(String(500), nullable=True)
    metrics = Column(JSON, nullable=True)
    error_message = Column(String(1000), nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("Users", lazy="joined")
    predictions = relationship("Predictions", back_populates="pipeline_run", lazy="select")
    deployments = relationship("DeployedModels", back_populates="pipeline_run", lazy="select")
