from services.pipelines.feature_strategies.base import FeatureStrategy
from services.pipelines.feature_strategies.heart_disease_features import HeartDiseaseFeatures

STRATEGY_REGISTRY: dict[str, type[FeatureStrategy]] = {
    "heart_disease": HeartDiseaseFeatures,
}
