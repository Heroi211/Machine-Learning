import pandas as pd
import numpy as np
import logging
from services.pipelines.feature_strategies.base import FeatureStrategy

logger = logging.getLogger(__name__)

EPS = 1


class HeartDiseaseFeatures(FeatureStrategy):
    """
    Features específicas para o domínio de doenças cardíacas.
    Baseadas em variáveis clínicas: idade, colesterol, frequência cardíaca,
    pressão arterial, depressão ST, entre outras.
    """

    def required_columns(self) -> list[str]:
        return ["age", "chol"]

    def created_features(self) -> list[str]:
        return [
            "age_squared",
            "cholesterol_to_age",
            "max_hr_pct",
            "bp_chol_ratio",
            "fbs_flag",
            "exang_flag",
            "stress_index",
            "age_decade",
            "risk_interaction",
            "high_st_depression_flag",
        ]

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cols = set(out.columns)

        out["age_squared"] = out["age"] ** 2

        out["cholesterol_to_age"] = out["chol"] / (out["age"] + EPS)

        if {"thalch", "age"} <= cols:
            predicted_max_hr = (220 - out["age"]).clip(lower=1)
            out["max_hr_pct"] = out["thalch"] / (predicted_max_hr + EPS)

        if {"trestbps", "chol"} <= cols:
            out["bp_chol_ratio"] = out["trestbps"] / (out["chol"] + EPS)

        fbs_col = "fbs" if "fbs" in cols else "fbs_True" if "fbs_True" in cols else None
        if fbs_col:
            out["fbs_flag"] = out[fbs_col].astype(int)

        exang_col = "exang" if "exang" in cols else "exang_True" if "exang_True" in cols else None
        if exang_col:
            out["exang_flag"] = out[exang_col].astype(int)

        if {"thalch", "trestbps"} <= cols:
            out["stress_index"] = out["thalch"] / (out["trestbps"] + EPS)

        out["age_decade"] = (out["age"] // 10).astype(int)

        if {"age", "oldpeak"} <= cols:
            out["risk_interaction"] = out["age"] * out["oldpeak"]

        if "oldpeak" in cols:
            out["high_st_depression_flag"] = (out["oldpeak"] > 1.0).astype(int)

        created = [f for f in self.created_features() if f in out.columns]
        logger.info(f"HeartDiseaseFeatures — {len(created)} features criadas: {created}")

        return out
