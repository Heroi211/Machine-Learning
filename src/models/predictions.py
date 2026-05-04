"""Modelo ORM para registros de predições realizadas pela API."""

from sqlalchemy import Column, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from core.generic import modelsGeneric


class Predictions(modelsGeneric):
    """Representa uma predição persistida com entrada, saída e probabilidade."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pipeline_run_id = Column(Integer, ForeignKey("pipeline_runs.id"), nullable=False)
    input_data = Column(JSON, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=True)

    user = relationship("Users", lazy="joined")
    pipeline_run = relationship("PipelineRuns", back_populates="predictions", lazy="joined")
