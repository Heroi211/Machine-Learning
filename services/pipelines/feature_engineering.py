import logging
import os
import sys
import time
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import joblib

from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform

from core.configs import settings
from core.custom_logger import setup_log
from services.pipelines.feature_strategies.base import FeatureStrategy
from services.pipelines.fe_hyperparameter_tuning import build_fresh_tuning_pipeline, param_distributions_for
from services.pipelines.fe_model_selection import normalize_optimization_metric, result_column_for_metric, select_best_model_after_training, sklearn_scoring_parameter, test_set_score

os.makedirs(settings.mlflow_artifact_root, exist_ok=True)
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

logger = logging.getLogger("ml.pipeline")

plt.style.use("seaborn-v0_8-darkgrid")


class FeatureEngineering:
    """
    Pipeline de Feature Engineering, seleção de features,
    treinamento comparativo, tuning e persistência.
    """

    def __init__(self, objective: str, strategy: FeatureStrategy, run_timestamp: str | None = None, csv_path: str | None = None, optimization_metric: str = "accuracy"):
        self.objective = objective
        self.strategy = strategy
        self._explicit_csv_path = os.path.abspath(csv_path) if csv_path else None
        self.optimization_metric = normalize_optimization_metric(optimization_metric)

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

        self.trained_models: dict = {}
        self.results_df: pd.DataFrame | None = None
        self.best_model_name: str | None = None
        self.best_pipeline = None
        self.best_params: dict | None = None
        self.best_cv_score: float = -np.inf
        self.best_test_score: float = -np.inf
        self.tuned_metrics: dict = {}
        self.figs_to_log: list[tuple[str, plt.Figure]] = []
        self.n_jobs = 1 if "debugpy" in sys.modules else -1

    # ------------------------------------------------------------------
    # Etapa 1 — Carregar dados pré-processados (saída do baseline)
    # ------------------------------------------------------------------

    def load_data(self):
        logger.info("Carregando dataset pré-processado...")

        if self._explicit_csv_path:
            if not os.path.isfile(self._explicit_csv_path):
                raise ValueError(f"CSV não encontrado: {self._explicit_csv_path}")
            file_must_modern = self._explicit_csv_path
            logger.info(f"Arquivo CSV (rota explícita): {file_must_modern}")
        else:
            csv_pattern = os.path.join(self.path_data_preprocessed, "*.csv")
            files = glob.glob(csv_pattern)
            if not files:
                raise ValueError(f"Nenhum CSV encontrado em {self.path_data_preprocessed}")

            file_must_modern = max(files, key=os.path.getctime)
            logger.info(f"Arquivo selecionado (modo legado — último ctime): {file_must_modern}")

        df = pd.read_csv(file_must_modern)

        has_prefixes = any("__" in col for col in df.columns)
        if has_prefixes:
            df.columns = [col.split("__", 1)[-1] if "__" in col else col for col in df.columns]
            logger.warning("Prefixos de colunas detectados e removidos (CSV legado).")

        logger.info(f"Colunas: {df.columns.tolist()}")

        target_column = df.columns[-1]
        if target_column != "target":
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
        logger.info("Iniciando split e seleção de features...")

        y = self.data["target"]
        x = self.data.drop(columns=["target"])
        self.feature_names = x.columns.tolist()

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        k_actual = min(k, self.x_train.shape[1])
        selector = SelectKBest(f_classif, k=k_actual)
        selector.fit(self.x_train, self.y_train)

        mask = selector.get_support()
        selected = [n for n, keep in zip(self.feature_names, mask) if keep]

        self.x_train = self.x_train[selected]
        self.x_test = self.x_test[selected]
        self.feature_names = selected

        logger.info(f"Features selecionadas ({len(selected)}):")
        for i, feat in enumerate(selected, 1):
            logger.info(f"  {i}. {feat}")

    # ------------------------------------------------------------------
    # Etapa 4 — Treinamento comparativo
    # ------------------------------------------------------------------

    def train_models(self):
        sort_col = result_column_for_metric(self.optimization_metric)
        logger.info(
            "Iniciando comparação de modelos (métrica de otimização: %s → coluna %s)",
            self.optimization_metric,
            sort_col,
        )

        model_configs = {
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "SVM": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=self.random_state),
        }

        results = []

        for name, estimator in model_configs.items():
            steps = []
            if name == "SVM":
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", estimator))

            pipeline = SkPipeline(steps)
            pipeline.fit(self.x_train, self.y_train)
            y_pred = pipeline.predict(self.x_test)

            metrics = {
                "Modelo": name,
                "Acurácia": accuracy_score(self.y_test, y_pred),
                "Precisão": precision_score(self.y_test, y_pred, zero_division=0),
                "Recall": recall_score(self.y_test, y_pred, zero_division=0),
                "F1": f1_score(self.y_test, y_pred, zero_division=0),
                "ROC AUC": np.nan,
            }

            if hasattr(pipeline, "predict_proba"):
                metrics["ROC AUC"] = roc_auc_score(self.y_test, pipeline.predict_proba(self.x_test)[:, 1])
            elif hasattr(pipeline, "decision_function"):
                metrics["ROC AUC"] = roc_auc_score(self.y_test, pipeline.decision_function(self.x_test))

            logger.info("=== %s === %s: %s", name, sort_col, metrics[sort_col])
            logger.debug(f"\n{classification_report(self.y_test, y_pred)}")

            self.trained_models[name] = pipeline
            results.append(metrics)

        self.results_df = (
            pd.DataFrame(results).sort_values(sort_col, ascending=False).reset_index(drop=True).round(4)
        )
        logger.info(f"Ranking de modelos:\n{self.results_df.to_string()}")

        winner_name, winner_score, _ = select_best_model_after_training(results, self.optimization_metric)
        self.best_model_name = winner_name
        self.best_pipeline = self.trained_models[winner_name]
        logger.info(
            "Modelo selecionado para a etapa seguinte: %s (%s=%.4f)",
            winner_name,
            sort_col,
            winner_score,
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

    def tune(self, time_limit_minutes: int = 60, acc_target: float = 0.90):
        sort_col = result_column_for_metric(self.optimization_metric)
        scoring = sklearn_scoring_parameter(self.optimization_metric)
        if not self.best_model_name:
            raise RuntimeError("tune requer train_models prévio com best_model_name definido.")

        logger.info(
            "Iniciando tuning — modelo: %s | limite: %smin | métrica: %s | alvo em %s > %s",
            self.best_model_name,
            time_limit_minutes,
            self.optimization_metric,
            sort_col,
            acc_target,
        )

        start = time.time()
        deadline = start + time_limit_minutes * 60

        param_distributions = param_distributions_for(self.best_model_name)
        sampler = ParameterSampler(param_distributions, n_iter=1_000_000, random_state=self.random_state)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        self.best_test_score = -np.inf
        self.best_cv_score = -np.inf
        self.best_params = None
        tuned_pipeline: SkPipeline | None = None

        n_evaluated = 0

        for i, params in enumerate(sampler, start=1):
            now = time.time()
            if now >= deadline:
                logger.info("Tempo limite atingido — encerrando busca.")
                break

            pipeline = build_fresh_tuning_pipeline(self.best_model_name, self.random_state)
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

            pipeline.fit(self.x_train, self.y_train)
            y_pred_t = pipeline.predict(self.x_test)
            y_proba_t = pipeline.predict_proba(self.x_test) if hasattr(pipeline, "predict_proba") else None
            test_score = test_set_score(self.y_test, y_pred_t, y_proba_t, self.optimization_metric)

            if test_score > self.best_test_score:
                self.best_test_score = test_score
                self.best_cv_score = mean_cv
                self.best_params = params
                tuned_pipeline = pipeline
                elapsed = now - start
                logger.info(
                    "[%s] Novo melhor | %s_teste=%.4f | %s_cv=%.4f | t=%.1fm | params=%s",
                    i,
                    self.optimization_metric,
                    test_score,
                    self.optimization_metric,
                    mean_cv,
                    elapsed / 60,
                    params,
                )

            if i % 5 == 0:
                elapsed = now - start
                remaining = max(0.0, deadline - now)
                logger.debug(
                    "Iterações: %s | melhor_%s=%.4f | decorrido=%.1fm | restante=%.1fm",
                    i,
                    self.optimization_metric,
                    self.best_test_score,
                    elapsed / 60,
                    remaining / 60,
                )

            n_evaluated = i

        if tuned_pipeline is None:
            logger.warning(
                "Nenhuma iteração melhorou o modelo — mantendo %s da etapa comparativa.",
                self.best_model_name,
            )
            tuned_pipeline = self.best_pipeline
        else:
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
        logger.info("Melhor %s (teste): %.4f", sort_col, self.tuned_metrics[sort_col])
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
            with mlflow.start_run(run_name=f"fe_{self.objective}_{self.now}"):
                if self.best_params:
                    for k, v in self.best_params.items():
                        mlflow.log_param(k, str(v))

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

    def run(self, time_limit_minutes: int, acc_target: float):
        start_time = datetime.now()
        logger.info(f"Pipeline de Feature Engineering iniciado: {start_time}")

        self.load_data()
        logger.debug(f"Dados carregados: {datetime.now() - start_time}")

        self.build_features()
        logger.debug(f"Features criadas: {datetime.now() - start_time}")

        self.select_features()
        logger.debug(f"Features selecionadas: {datetime.now() - start_time}")

        self.train_models()
        logger.debug(f"Modelos treinados: {datetime.now() - start_time}")

        self.tune(time_limit_minutes=time_limit_minutes, acc_target=acc_target)
        logger.debug(f"Tuning concluído: {datetime.now() - start_time}")

        self.evaluate_importance()
        logger.debug(f"Importância avaliada: {datetime.now() - start_time}")

        self.save()

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
        pipeline.run()
    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {e}")
        raise
