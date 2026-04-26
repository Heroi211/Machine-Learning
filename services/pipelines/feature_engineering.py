import logging
import os
import sys
import time
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform

from core.configs import settings
from core.custom_logger import setup_log
from services.pipelines.feature_strategies.base import FeatureStrategy
from services.pipelines.fe_hyperparameter_tuning import param_distributions_for
from services.pipelines.fe_model_selection import normalize_optimization_metric, result_column_for_metric, sklearn_scoring_parameter

os.makedirs(settings.mlflow_artifact_root, exist_ok=True)
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

logger = logging.getLogger("ml.pipeline")

plt.style.use("seaborn-v0_8-darkgrid")


class FeatureEngineering:
    """
    Pipeline de Feature Engineering, seleção de features,
    treinamento comparativo, tuning e persistência.
    """

    def __init__(
        self,
        objective: str,
        strategy: FeatureStrategy,
        run_timestamp: str | None = None,
        csv_path: str | None = None,
        manifest_path: str | None = None,
        optimization_metric: str = "accuracy",
        min_precision: float | None = None,
        min_roc_auc: float | None = None,
        tuning_n_iter: int | None = None,
        export_figures_dir: str | None = None,
    ):
        self.objective = objective
        self.strategy = strategy
        self._explicit_csv_path = os.path.abspath(csv_path) if csv_path else None
        self._explicit_manifest_path = os.path.abspath(manifest_path) if manifest_path else None
        self.optimization_metric = normalize_optimization_metric(optimization_metric)
        self.min_precision = min_precision
        self.min_roc_auc = min_roc_auc
        self.export_figures_dir = export_figures_dir
        self.mlflow_run_id: str | None = None

        self.path_data_preprocessed = settings.path_data_preprocessed
        self.path_model = settings.path_model
        self.test_size = settings.test_size
        self.random_state = settings.random_state

        if run_timestamp is not None:
            self.now = run_timestamp
        else:
            self.now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.snapshot_path = os.path.join(settings.path_data, settings.path_logs, self.now)

        self.data: pd.DataFrame | None = None
        self.x_train: pd.DataFrame | None = None
        self.x_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.feature_names: list[str] = []
        self.feature_groups: dict[str, list[str]] = {"binary": [], "continuous": [], "categorical": []}
        self.k_features = 25

        self.trained_models: dict = {}
        self.baseline_manifest: dict | None = None
        self.results_df: pd.DataFrame | None = None
        self.guardrails_summary: dict = {}
        self.best_model_name: str | None = None
        self.best_pipeline = None
        self.best_params: dict | None = None
        self.best_cv_score: float = -np.inf
        self.best_test_score: float = -np.inf
        self.tuned_metrics: dict = {}
        self.figs_to_log: list[tuple[str, plt.Figure]] = []
        self.n_jobs = 1 if "debugpy" in sys.modules else -1
        self.tuning_n_iter = tuning_n_iter if tuning_n_iter is not None else 100

        for metric_name, metric_value in (
            ("min_precision", self.min_precision),
            ("min_roc_auc", self.min_roc_auc),
        ):
            if metric_value is not None and not (0.0 <= metric_value <= 1.0):
                raise ValueError(f"{metric_name} deve estar no intervalo [0, 1]. Recebido: {metric_value}")
        if self.tuning_n_iter <= 0:
            raise ValueError(f"tuning_n_iter deve ser > 0. Recebido: {self.tuning_n_iter}")

    def _passes_guardrails(self, precision_value: float, roc_auc_value: float) -> bool:
        if self.min_precision is not None and precision_value < self.min_precision:
            return False
        if self.min_roc_auc is not None:
            if np.isnan(roc_auc_value) or roc_auc_value < self.min_roc_auc:
                return False
        return True

    @staticmethod
    def _is_binary_column(series: pd.Series) -> bool:
        non_null = series.dropna()
        if non_null.empty:
            return False
        unique_values = set(non_null.unique().tolist())
        return unique_values.issubset({0, 1, 0.0, 1.0, np.int8(0), np.int8(1), True, False})

    def _classify_feature_columns(self, x_df: pd.DataFrame) -> dict[str, list[str]]:
        categorical_cols = x_df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = x_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

        binary_cols: list[str] = []
        continuous_cols: list[str] = []
        for col in numeric_cols:
            if self._is_binary_column(x_df[col]):
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        grouped = {"binary": binary_cols, "continuous": continuous_cols, "categorical": categorical_cols}
        self.feature_groups = grouped
        return grouped

    def _build_preprocess_transformer(
        self, x_train: pd.DataFrame, groups: dict[str, list[str]] | None = None
    ) -> ColumnTransformer:
        """
        Pré-processamento no mesmo padrão do Baseline:
        - binárias: passthrough (sem transformação)
        - contínuas: StandardScaler (+ imputação mediana só se houver nulos)
        - categóricas: OneHotEncoder (+ imputação moda só se houver nulos)
        """
        groups = groups or self._classify_feature_columns(x_train)
        binary_cols = groups["binary"]
        continuous_cols = groups["continuous"]
        categorical_cols = groups["categorical"]

        transformers: list[tuple] = []

        if continuous_cols:
            continuous_steps = []
            has_nulls_cont = bool(x_train[continuous_cols].isna().any().any())
            if has_nulls_cont:
                continuous_steps.append(("imputer", SimpleImputer(strategy="median")))
            continuous_steps.append(("scaler", StandardScaler()))
            transformers.append(("continuous", SkPipeline(continuous_steps), continuous_cols))

        if binary_cols:
            has_nulls_bin = bool(x_train[binary_cols].isna().any().any())
            if has_nulls_bin:
                raise ValueError(
                    "Colunas binárias com nulos detectadas. "
                    "Como binárias estão em passthrough por definição, "
                    "preencha nulos antes do treino."
                )
            transformers.append(("binary", "passthrough", binary_cols))

        if categorical_cols:
            categorical_steps = []
            has_nulls_cat = bool(x_train[categorical_cols].isna().any().any())
            if has_nulls_cat:
                categorical_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            categorical_steps.append(
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
            )
            transformers.append(("categorical", SkPipeline(categorical_steps), categorical_cols))

        if not transformers:
            raise ValueError("Nenhuma coluna válida encontrada para pré-processamento no FE.")

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _build_model_pipeline(self, model_name: str):
        preprocess = self._build_preprocess_transformer(self.x_train, groups=self.feature_groups)
        model_map = {
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "SVM": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
        }
        if model_name not in model_map:
            raise ValueError(f"Modelo não suportado no FE: {model_name}")
        return SkPipeline(
            [
                ("preprocess", preprocess),
                ("selector", SelectKBest(f_classif, k=self.k_features)),
                ("model", model_map[model_name]),
            ]
        )

    # ------------------------------------------------------------------
    # Etapa 1 — Carregar dados pré-processados (saída do baseline)
    # ------------------------------------------------------------------

    def load_data(self):
        logger.info("Carregando dataset pré-processado...")
        manifest_path = self._explicit_manifest_path or os.path.join(self.path_data_preprocessed, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise ValueError(f"Manifest não encontrado: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.baseline_manifest = manifest

        if manifest.get("objective") != self.objective:
            raise ValueError(
                f"Manifest do baseline com objective='{manifest.get('objective')}', "
                f"mas FE foi chamado para '{self.objective}'."
            )
        if manifest.get("sample_schema") != "raw_clean":
            raise ValueError(
                f"sample_schema inválido no manifest: {manifest.get('sample_schema')!r}. "
                "Esperado: 'raw_clean'."
            )

        file_from_manifest = manifest.get("output_sample_csv_stable")
        if not file_from_manifest:
            raise ValueError("Manifest inválido: campo 'output_sample_csv_stable' ausente.")

        csv_path = os.path.abspath(file_from_manifest)
        if self._explicit_csv_path and os.path.abspath(self._explicit_csv_path) != csv_path:
            logger.warning(
                "csv_path explícito (%s) difere do manifest (%s). Usando manifest como fonte de verdade.",
                self._explicit_csv_path,
                csv_path,
            )
        if not os.path.isfile(csv_path):
            raise ValueError(f"CSV do baseline não encontrado (manifest): {csv_path}")

        logger.info(f"CSV selecionado via manifest: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()

        has_prefixes = any("__" in col for col in df.columns)
        if has_prefixes:
            df.columns = [col.split("__", 1)[-1] if "__" in col else col for col in df.columns]
            logger.warning("Prefixos de colunas detectados e removidos (CSV legado).")

        logger.info(f"Colunas: {df.columns.tolist()}")

        if not 'target' in df.columns:
            logger.error(f"Coluna 'target' não encontrada no dataset pré-processado no final do CSV.")
            raise ValueError("Pipeline interrompido — coluna 'target' não encontrada no dataset pré-processado.")
        
        null_total = df.isnull().sum().sum()
        if null_total > 0:
            logger.error(f"Valores nulos encontrados: {null_total}")
            raise ValueError("Pipeline interrompido — valores nulos no dataset pré-processado.")

        logger.info(f"Dataset carregado: {df.shape}")
        logger.info(f"Target:\n{df['target'].value_counts().to_string()}")

        self.data = df

    # ------------------------------------------------------------------
    # Etapa 2 — Criar features (delega para a strategy)
    # ------------------------------------------------------------------

    def build_features(self):
        logger.info(f"Construindo features com strategy: {self.strategy.__class__.__name__}")

        self.strategy.validate(self.data)
        self.data = self.strategy.build(self.data)

        logger.info(f"Dataset após feature engineering: {self.data.shape}")

    # ------------------------------------------------------------------
    # Etapa 3 — Split + Seleção de features
    # ------------------------------------------------------------------

    def select_features(self, k: int = 25):
        logger.info("Iniciando split de dados para FE...")

        y = self.data["target"]
        x = self.data.drop(columns=["target"])
        self.feature_names = x.columns.tolist()
        self.k_features = min(k, max(1, x.shape[1]))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        groups = self._classify_feature_columns(self.x_train)
        logger.info(
            "Feature groups (pré-build): binárias=%s contínuas=%s categóricas=%s",
            len(groups["binary"]),
            len(groups["continuous"]),
            len(groups["categorical"]),
        )
        logger.debug("Colunas binárias: %s", groups["binary"])
        logger.debug("Colunas contínuas: %s", groups["continuous"])
        logger.debug("Colunas categóricas: %s", groups["categorical"])
        logger.info("k de seleção configurado para SelectKBest: %s", self.k_features)

    # ------------------------------------------------------------------
    # Etapa 4 — Treinamento comparativo
    # ------------------------------------------------------------------

    def train_models(self):
        sort_col = result_column_for_metric(self.optimization_metric)
        scoring = sklearn_scoring_parameter(self.optimization_metric)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        logger.info(
            "Iniciando comparação de modelos (métrica de otimização: %s → coluna %s)",
            self.optimization_metric,
            sort_col,
        )

        model_configs = [
            "Decision Tree",
            "Random Forest",
            "SVM",
            "Gradient Boosting",
        ]

        results = []

        for name in model_configs:
            pipeline = self._build_model_pipeline(name)
            cv_scores = cross_val_score(
                pipeline, self.x_train, self.y_train, cv=cv, scoring=scoring, n_jobs=self.n_jobs
            )
            cv_score = float(np.mean(cv_scores))

            cv_guard = cross_validate(
                pipeline,
                self.x_train,
                self.y_train,
                cv=cv,
                scoring={"precision": "precision", "roc_auc": "roc_auc"},
                n_jobs=self.n_jobs,
            )
            guardrail_precision_cv = float(np.mean(cv_guard["test_precision"]))
            guardrail_roc_auc_cv = float(np.mean(cv_guard["test_roc_auc"]))

            pipeline.fit(self.x_train, self.y_train)
            y_pred = pipeline.predict(self.x_test)

            metrics = {
                "Modelo": name,
                "Acurácia": accuracy_score(self.y_test, y_pred),
                "Precisão": precision_score(self.y_test, y_pred, zero_division=0),
                "Recall": recall_score(self.y_test, y_pred, zero_division=0),
                "F1": f1_score(self.y_test, y_pred, zero_division=0),
                "ROC AUC": np.nan,
                "CV Score": cv_score,
            }

            if hasattr(pipeline, "predict_proba"):
                metrics["ROC AUC"] = roc_auc_score(self.y_test, pipeline.predict_proba(self.x_test)[:, 1])
            elif hasattr(pipeline, "decision_function"):
                metrics["ROC AUC"] = roc_auc_score(self.y_test, pipeline.decision_function(self.x_test))

            metrics["Pass Guardrails"] = self._passes_guardrails(
                guardrail_precision_cv,
                guardrail_roc_auc_cv,
            )
            metrics["CV Precision"] = guardrail_precision_cv
            metrics["CV ROC AUC"] = guardrail_roc_auc_cv

            logger.info("=== %s === cv_%s: %.4f", name, self.optimization_metric, cv_score)
            logger.debug(f"\n{classification_report(self.y_test, y_pred)}")

            self.trained_models[name] = pipeline
            results.append(metrics)

        self.results_df = pd.DataFrame(results).sort_values("CV Score", ascending=False).reset_index(drop=True).round(4)
        logger.info(f"Ranking de modelos:\n{self.results_df.to_string()}")

        eligible = self.results_df[self.results_df["Pass Guardrails"] == True]
        if not eligible.empty:
            winner_row = eligible.iloc[0]
            self.guardrails_summary["selection_guardrails_passed"] = True
        else:
            winner_row = self.results_df.iloc[0]
            self.guardrails_summary["selection_guardrails_passed"] = False
            logger.warning(
                "Nenhum modelo passou nos guardrails (min_precision=%s, min_roc_auc=%s). "
                "Selecionando melhor modelo por cv_%s.",
                self.min_precision,
                self.min_roc_auc,
                self.optimization_metric,
            )
        self.best_model_name = str(winner_row["Modelo"])
        self.best_pipeline = self.trained_models[self.best_model_name]
        self.best_cv_score = float(winner_row["CV Score"])
        logger.info(
            "Modelo selecionado para a etapa seguinte: %s (cv_%s=%.4f)",
            self.best_model_name,
            self.optimization_metric,
            self.best_cv_score,
        )

    # ------------------------------------------------------------------
    # Etapa 5 — Tuning (time-budget)
    # ------------------------------------------------------------------

    def _finalize_tuned_metrics(self, y_pred, y_proba_vec: np.ndarray | None) -> None:
        """Preenche `tuned_metrics` a partir de predições no conjunto de teste (`y_proba_vec` = P(classe positiva))."""
        self.tuned_metrics = {
            "Acurácia": accuracy_score(self.y_test, y_pred),
            "Precisão": precision_score(self.y_test, y_pred, zero_division=0),
            "Recall": recall_score(self.y_test, y_pred, zero_division=0),
            "F1": f1_score(self.y_test, y_pred, zero_division=0),
            "ROC AUC": (
                roc_auc_score(self.y_test, y_proba_vec)
                if y_proba_vec is not None and not np.all(y_proba_vec == 0)
                else np.nan
            ),
        }

    def tune(self, time_limit_minutes: int = 60, acc_target: float | None = None):
        sort_col = result_column_for_metric(self.optimization_metric)
        scoring = sklearn_scoring_parameter(self.optimization_metric)
        if not self.best_model_name:
            raise RuntimeError("tune requer train_models prévio com best_model_name definido.")

        if acc_target is not None:
            logger.info(
                "Iniciando tuning — modelo: %s | limite: %smin | métrica: %s | alvo em %s > %s",
                self.best_model_name,
                time_limit_minutes,
                self.optimization_metric,
                sort_col,
                acc_target,
            )
        else:
            logger.info(
                "Iniciando tuning — modelo: %s | limite: %smin | métrica: %s | sem alvo explícito",
                self.best_model_name,
                time_limit_minutes,
                self.optimization_metric,
            )

        start = time.time()
        deadline = start + time_limit_minutes * 60

        param_distributions = param_distributions_for(self.best_model_name)
        sampler = ParameterSampler(param_distributions, n_iter=self.tuning_n_iter, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        self.best_cv_score = -np.inf
        self.best_params = None
        tuned_pipeline: SkPipeline | None = None
        tuned_pipeline_fallback: SkPipeline | None = None
        best_cv_fallback = -np.inf
        best_params_fallback: dict | None = None

        n_evaluated = 0

        for i, params in enumerate(sampler, start=1):
            now = time.time()
            if now >= deadline:
                logger.info("Tempo limite atingido — encerrando busca.")
                break

            pipeline = self._build_model_pipeline(self.best_model_name)
            pipeline.set_params(**params)

            cv_scores = cross_val_score(
                pipeline,
                self.x_train,
                self.y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
            )
            mean_cv = float(np.mean(cv_scores))

            guardrail_precision_cv = float("nan")
            guardrail_roc_auc_cv = float("nan")
            if self.min_precision is not None or self.min_roc_auc is not None:
                scorers = {}
                if self.min_precision is not None:
                    scorers["precision"] = "precision"
                if self.min_roc_auc is not None:
                    scorers["roc_auc"] = "roc_auc"
                cv_guard = cross_validate(
                    pipeline,
                    self.x_train,
                    self.y_train,
                    cv=cv,
                    scoring=scorers,
                    n_jobs=self.n_jobs,
                )
                if "test_precision" in cv_guard:
                    guardrail_precision_cv = float(np.mean(cv_guard["test_precision"]))
                if "test_roc_auc" in cv_guard:
                    guardrail_roc_auc_cv = float(np.mean(cv_guard["test_roc_auc"]))

            pass_guardrails = self._passes_guardrails(guardrail_precision_cv, guardrail_roc_auc_cv)

            pipeline.fit(self.x_train, self.y_train)

            if mean_cv > best_cv_fallback:
                best_cv_fallback = mean_cv
                best_params_fallback = params
                tuned_pipeline_fallback = pipeline

            if pass_guardrails and mean_cv > self.best_cv_score:
                self.best_cv_score = mean_cv
                self.best_params = params
                tuned_pipeline = pipeline
                elapsed = now - start
                logger.info(
                    "[%s] Novo melhor (guardrails ok) | cv_%s=%.4f | cv_precision=%.4f | cv_roc_auc=%.4f | t=%.1fm | params=%s",
                    i,
                    self.optimization_metric,
                    mean_cv,
                    guardrail_precision_cv,
                    guardrail_roc_auc_cv,
                    elapsed / 60,
                    params,
                )

            if i % 5 == 0:
                elapsed = now - start
                remaining = max(0.0, deadline - now)
                logger.debug(
                    "Iterações: %s | melhor_cv_%s=%.4f | decorrido=%.1fm | restante=%.1fm",
                    i,
                    self.optimization_metric,
                    self.best_cv_score,
                    elapsed / 60,
                    remaining / 60,
                )

            n_evaluated = i

        if tuned_pipeline is None:
            self.guardrails_summary["tuning_guardrails_passed"] = False
            if tuned_pipeline_fallback is not None:
                logger.warning(
                    "Nenhuma iteração do tuning passou nos guardrails (min_precision=%s, min_roc_auc=%s). "
                    "Usando melhor configuração por cv_%s sem guardrails.",
                    self.min_precision,
                    self.min_roc_auc,
                    self.optimization_metric,
                )
                tuned_pipeline = tuned_pipeline_fallback
                self.best_params = best_params_fallback
                self.best_cv_score = best_cv_fallback
            else:
                logger.warning(
                    "Nenhuma iteração melhorou o modelo — mantendo %s da etapa comparativa.",
                    self.best_model_name,
                )
                tuned_pipeline = self.best_pipeline
        else:
            self.guardrails_summary["tuning_guardrails_passed"] = True

        self.best_pipeline = tuned_pipeline

        y_pred = self.best_pipeline.predict(self.x_test)
        y_proba = (
            self.best_pipeline.predict_proba(self.x_test)[:, 1]
            if hasattr(self.best_pipeline, "predict_proba")
            else None
        )
        self._finalize_tuned_metrics(y_pred, y_proba)

        elapsed_total = time.time() - start
        logger.info(f"Tuning concluído — {n_evaluated} iterações em {elapsed_total / 60:.1f}min")
        logger.info("Melhor cv_%s: %.4f", self.optimization_metric, self.best_cv_score)
        if acc_target is not None:
            logger.info(
                "Alvo (%.2f) sobre %s: %s",
                acc_target,
                sort_col,
                "atingido" if self.tuned_metrics[sort_col] > acc_target else "não atingido",
            )

    # ------------------------------------------------------------------
    # Etapa 6 — Importância de features
    # ------------------------------------------------------------------

    def evaluate_importance(self):
        logger.info("Calculando importância de features...")

        feat_names = np.array(self.feature_names)
        if hasattr(self.best_pipeline, "named_steps") and "preprocess" in self.best_pipeline.named_steps:
            preprocess = self.best_pipeline.named_steps["preprocess"]
            feat_names = np.array(preprocess.get_feature_names_out())
        if hasattr(self.best_pipeline, "named_steps") and "selector" in self.best_pipeline.named_steps:
            selector = self.best_pipeline.named_steps["selector"]
            if hasattr(selector, "get_support"):
                feat_names = feat_names[selector.get_support()]

        model_step = (
            self.best_pipeline.named_steps.get("model", self.best_pipeline)
            if hasattr(self.best_pipeline, "named_steps")
            else self.best_pipeline
        )

        tree_imp = getattr(model_step, "feature_importances_", None)
        if tree_imp is not None and len(tree_imp) == len(feat_names):
            imp_df = pd.DataFrame({"feature": feat_names, "importance": tree_imp}).sort_values("importance", ascending=False)
            logger.info(f"Importância Gini (top 20):\n{imp_df.head(20).to_string()}")

            fig, ax = plt.subplots(figsize=(8, min(0.45 * len(imp_df.head(20)), 10)))
            sns.barplot(data=imp_df.head(20), x="importance", y="feature", color="#1f77b4", ax=ax)
            ax.set_title("Importância de Features (Gini) — Top 20")
            ax.set_xlabel("Importância")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            if self.export_figures_dir:
                os.makedirs(self.export_figures_dir, exist_ok=True)
                gini_png = os.path.join(self.export_figures_dir, "feature_importance_gini_top20.png")
                fig.savefig(gini_png, dpi=200, bbox_inches="tight")
            self.figs_to_log.append(("feature_importance_gini_top20.png", fig))
            plt.close(fig)
        else:
            logger.info("Modelo não expõe feature_importances_. Pulando gráfico Gini.")

        perm = permutation_importance(
            self.best_pipeline,
            self.x_test,
            self.y_test,
            scoring=sklearn_scoring_parameter(self.optimization_metric),
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        perm_df = pd.DataFrame({
            "feature": feat_names,
            "importance": perm.importances_mean,
            "std": perm.importances_std,
        }).sort_values("importance", ascending=False)
        logger.info(f"Importância por permutação (top 20):\n{perm_df.head(20).to_string()}")

        fig2, ax2 = plt.subplots(figsize=(8, min(0.45 * len(perm_df.head(20)), 10)))
        sns.barplot(data=perm_df.head(20), x="importance", y="feature", color="#ff7f0e", ax=ax2)
        ax2.set_title(f"Importância por Permutação ({self.optimization_metric}) — Top 20")
        ax2.set_xlabel("Queda média na métrica")
        ax2.set_ylabel("Feature")
        plt.tight_layout()
        if self.export_figures_dir:
            os.makedirs(self.export_figures_dir, exist_ok=True)
            p_png = os.path.join(self.export_figures_dir, "feature_importance_permutation_top20.png")
            fig2.savefig(p_png, dpi=200, bbox_inches="tight")
        self.figs_to_log.append(("feature_importance_permutation_top20.png", fig2))
        plt.close(fig2)

    # ------------------------------------------------------------------
    # Etapa 7 — Persistência (joblib + MLflow)
    # ------------------------------------------------------------------

    def save(self):
        logger.info("Salvando artefatos...")

        os.makedirs(self.path_model, exist_ok=True)

        joblib_path = os.path.join(self.path_model, f"best_{self.objective}_{self.now}.joblib")
        joblib.dump(self.best_pipeline, joblib_path)
        logger.info(f"Modelo salvo em: {joblib_path}")

        try:
            experiment_name = f"{self.objective}_feature_engineering"
            if not mlflow.get_experiment_by_name(experiment_name):
                mlflow.create_experiment(experiment_name, artifact_location=settings.mlflow_artifact_root)
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=f"fe_{self.objective}_{self.now}") as _mfe:
                self.mlflow_run_id = _mfe.info.run_id
                if self.best_params:
                    for k, v in self.best_params.items():
                        mlflow.log_param(k, str(v))
                mlflow.log_param("optimization_metric", self.optimization_metric)
                mlflow.log_param("tuning_n_iter", self.tuning_n_iter)
                if self.min_precision is not None:
                    mlflow.log_param("min_precision", float(self.min_precision))
                if self.min_roc_auc is not None:
                    mlflow.log_param("min_roc_auc", float(self.min_roc_auc))
                for k, v in self.guardrails_summary.items():
                    mlflow.log_param(k, bool(v))
                if self.baseline_manifest is not None:
                    mlflow.log_dict(self.baseline_manifest, "baseline_manifest.json")

                if np.isfinite(self.best_cv_score):
                    mlflow.log_metric(f"cv_{self.optimization_metric}", float(self.best_cv_score))
                for k, v in self.tuned_metrics.items():
                    if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                        mlflow.log_metric(k.replace(" ", "_").lower(), float(v))

                for fname, fobj in self.figs_to_log:
                    try:
                        mlflow.log_figure(fobj, fname)
                    except Exception as e:
                        logger.warning(f"Falha ao logar figura {fname}: {e}")

                mlflow.sklearn.log_model(self.best_pipeline, artifact_path="sklearn_model")
                mlflow.log_artifact(joblib_path, artifact_path="joblib")

            logger.info(f"Resultados registrados no MLflow (experimento: {experiment_name})")
        except Exception as e:
            logger.error(f"Falha ao registrar no MLflow: {e}")

    # ------------------------------------------------------------------
    # Orquestrador
    # ------------------------------------------------------------------

    def _run_data_contract_and_fe_build(self, start_time):
        """
        Responsabilidade: carregar contrato do baseline e construir features.
        """
        self.load_data()
        logger.debug(f"Dados carregados: {datetime.now() - start_time}")

        self.build_features()
        logger.debug(f"Features criadas: {datetime.now() - start_time}")

    def _run_modeling_prep_and_selection(self, start_time):
        """
        Responsabilidade: preparar split e selecionar modelo base.
        """
        self.select_features()
        logger.debug(f"Features selecionadas: {datetime.now() - start_time}")

        self.train_models()
        logger.debug(f"Modelos treinados: {datetime.now() - start_time}")

    def _run_tuning_evaluation_and_persistence(self, start_time, time_limit_minutes: int, acc_target: float | None):
        """
        Responsabilidade: tuning, avaliação final e persistência.
        """
        self.tune(time_limit_minutes=time_limit_minutes, acc_target=acc_target)
        logger.debug(f"Tuning concluído: {datetime.now() - start_time}")

        self.evaluate_importance()
        logger.debug(f"Importância avaliada: {datetime.now() - start_time}")

        self.save()
        logger.debug(f"Artefatos salvos: {datetime.now() - start_time}")

    def run(self, time_limit_minutes: int, acc_target: float | None):
        """
        Orquestrador principal: executa FE por responsabilidades.
        """
        start_time = datetime.now()
        logger.info(f"Pipeline de Feature Engineering iniciado: {start_time}")

        self._run_data_contract_and_fe_build(start_time)
        self._run_modeling_prep_and_selection(start_time)
        self._run_tuning_evaluation_and_persistence(start_time, time_limit_minutes, acc_target)

        elapsed = datetime.now() - start_time
        logger.info(f"Pipeline de Feature Engineering concluído em: {elapsed}")


if __name__ == "__main__":
    from services.pipelines.feature_strategies import STRATEGY_REGISTRY

    objective = "heart_disease"

    if objective not in STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{objective}' não registrada. Disponíveis: {list(STRATEGY_REGISTRY.keys())}")

    strategy = STRATEGY_REGISTRY[objective]()

    snapshot_path = os.path.join(settings.path_data, settings.path_logs, datetime.now().strftime("%Y%m%d_%H%M%S"))
    setup_log(snapshot_path, datetime.now().strftime("%Y%m%d_%H%M%S"))

    pipeline = FeatureEngineering(objective=objective, strategy=strategy)

    try:
        pipeline.run(time_limit_minutes=2, acc_target=None)
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}")
        raise
