"""Importa todos os modelos ORM para que o SQLAlchemy registre os relacionamentos."""

from models.roles import Roles
from models.users import Users
from models.pipeline_runs import PipelineRuns
from models.predictions import Predictions
from models.deployed_models import DeployedModels
