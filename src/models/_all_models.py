"""Importa todos os modelos ORM para facilitar registro e descoberta."""

from models.roles import Roles
from models.users import Users
from models.pipeline_runs import PipelineRuns
from models.predictions import Predictions
from models.deployed_models import DeployedModels

__all__ = [
    "Roles",
    "Users",
    "PipelineRuns",
    "Predictions",
    "DeployedModels",
]
