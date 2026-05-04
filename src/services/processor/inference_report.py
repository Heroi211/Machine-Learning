"""
Relatório de inferência para respostas de ``/predict``.

Centraliza leitura de ``PipelineRuns.metrics`` (gravadas no FE) para o consumidor
saber **o que está a ser servido** (sklearn vs MLP) e **métricas de holdout do treino**
(≠ métricas desta linha de predição).
"""

from __future__ import annotations

import math
from typing import Any

from schemas.processor_schemas import InferenceReport, MetricSnapshot


def attach_mlp_metrics_snapshot(merged_metrics: dict, pipeline: Any) -> None:
    """Acrescenta métricas de teste/validação do MLP a ``merged_metrics`` se o treino as tiver."""
    res = getattr(pipeline, "mlp_torch_result", None)
    if res is None:
        return
    merged_metrics["mlp_metrics_test"] = {k: float(v) for k, v in res.metrics_test.items()}
    merged_metrics["mlp_metrics_val"] = {k: float(v) for k, v in res.metrics_val.items()}
    training: dict[str, Any] = {
        "best_epoch": int(res.best_epoch),
        "best_val_loss": float(res.best_val_loss),
    }
    hp = getattr(pipeline, "mlp_torch_hparams", None) or {}
    for key in (
        "hidden_dims",
        "dropout",
        "batch_size",
        "lr",
        "weight_decay",
        "max_epochs",
        "early_stopping_patience",
        "val_fraction",
    ):
        if key not in hp:
            continue
        val = hp[key]
        if key == "hidden_dims" and isinstance(val, (tuple, list)):
            training[key] = [int(x) for x in val]
        else:
            training[key] = val
    merged_metrics["mlp_training"] = training


def attach_fe_model_comparison_table(merged_metrics: dict, pipeline: Any) -> None:
    """Serializa a tabela sklearn pré/pós-tuning + MLP (teste) tal como no ``fe_export``."""
    builder = getattr(pipeline, "_build_model_comparison_table", None)
    if not callable(builder):
        return
    df = builder()
    if df is None or getattr(df, "empty", True):
        return
    rows = []
    for rec in df.round(6).to_dict(orient="records"):
        row = {}
        for k, v in rec.items():
            if v is None:
                row[k] = None
            elif isinstance(v, (float, int)) and isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
            elif isinstance(v, (float, int)):
                row[k] = float(v) if isinstance(v, float) else int(v)
            else:
                row[k] = str(v)
        rows.append(row)
    merged_metrics["fe_model_comparison_table"] = rows


def _snapshot_from_sklearn_tuned(m: dict) -> MetricSnapshot | None:
    if not any(k in m for k in ("Acurácia", "Precisão", "Recall", "F1", "ROC AUC")):
        return None

    def _f(key: str) -> float | None:
        v = m.get(key)
        if v is None:
            return None
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    return MetricSnapshot(
        accuracy=_f("Acurácia"),
        precision=_f("Precisão"),
        recall=_f("Recall"),
        f1=_f("F1"),
        roc_auc=_f("ROC AUC"),
    )


def _snapshot_from_mlp_test(d: Any) -> MetricSnapshot | None:
    if not isinstance(d, dict) or not d:
        return None

    def _g(key: str) -> float | None:
        v = d.get(key)
        if v is None:
            return None
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    return MetricSnapshot(
        accuracy=_g("accuracy"),
        precision=_g("precision"),
        recall=_g("recall"),
        f1=_g("f1"),
        roc_auc=_g("roc_auc"),
    )


def _snapshot_from_baseline_run_metrics(br: dict) -> MetricSnapshot | None:
    """Métricas de teste do pipeline Baseline (chaves ``test_*`` na BD)."""

    def _g(*keys: str) -> float | None:
        for key in keys:
            v = br.get(key)
            if v is None:
                continue
            try:
                x = float(v)
                return x if math.isfinite(x) else None
            except (TypeError, ValueError):
                continue
        return None

    acc = _g("test_accuracy")
    prec = _g("test_precision")
    rec = _g("test_recall")
    f1v = _g("test_f1")
    roc = _g("test_roc_auc")
    if all(x is None for x in (acc, prec, rec, f1v, roc)):
        return None
    return MetricSnapshot(accuracy=acc, precision=prec, recall=rec, f1=f1v, roc_auc=roc)


def build_inference_report(metrics: dict | None, inference_backend: str) -> InferenceReport:
    m = dict(metrics or {})
    backend = (inference_backend or m.get("inference_backend") or "sklearn").strip().lower()
    if backend not in ("sklearn", "mlp"):
        backend = "sklearn"

    predict_model = str(m.get("predict_model") or ("pytorch_mlp" if backend == "mlp" else "sklearn_pipeline"))
    sk_benchmark = m.get("sklearn_benchmark_classifier") or m.get("best_model_name")

    best_cv = m.get("best_cv_score")
    try:
        best_cv_f = float(best_cv) if best_cv is not None and math.isfinite(float(best_cv)) else None
    except (TypeError, ValueError):
        best_cv_f = None

    thr = m.get("classification_decision_threshold")
    try:
        thr_f = float(thr) if thr is not None else None
    except (TypeError, ValueError):
        thr_f = None

    sk_holdout = _snapshot_from_sklearn_tuned(m)
    mlp_holdout = _snapshot_from_mlp_test(m.get("mlp_metrics_test"))
    baseline_ref_raw = m.get("baseline_reference_metrics")
    baseline_ref = dict(baseline_ref_raw) if isinstance(baseline_ref_raw, dict) else None
    baseline_holdout = _snapshot_from_baseline_run_metrics(baseline_ref) if baseline_ref else None

    comp_raw = m.get("fe_model_comparison_table")
    fe_comparison: list[dict[str, Any]] = []
    if isinstance(comp_raw, list):
        fe_comparison = [dict(r) for r in comp_raw if isinstance(r, dict)]

    served = mlp_holdout if backend == "mlp" else sk_holdout
    notes: list[str] = []

    notes.append(
        "**Comparativo TC:** o **Baseline** usa sklearn (pipeline `Baseline`, p.ex. regressão logística). "
        "O **FE** acrescenta feature engineering, modelos sklearn avançados **e** a MLP PyTorch no mesmo run; "
        "o que é servido em `/predict` segue `inference_backend` (sklearn joblib ou MLP)."
    )

    if backend == "mlp":
        notes.append(
            "Inferência servida pela **MLP PyTorch** (bundle `.pt` + preprocess joblib). "
            "A probabilidade é σ(logit) da rede, alinhada ao pré-processador do FE."
        )
        if sk_holdout is None:
            notes.append(
                "Não há métricas de holdout do sklearn neste run (ex.: tuning sklearn desligado com "
                "`USE_MLP_FOR_PREDICTION=true`). O vencedor de CV (`sklearn_benchmark_classifier`) "
                "serve só como referência de estudo, não como modelo servido."
            )
        else:
            notes.append(
                "Bloco `sklearn_holdout_test` mostra métricas de **teste** do pipeline sklearn "
                "(após tuning quando existiu), para comparar com `served_holdout_metrics` da MLP."
            )
        if baseline_holdout:
            notes.append(
                "`baseline_holdout_metrics` veio do **PipelineRun de baseline activo** no momento do treino FE — "
                "mesma ideia de referência simples **antes** do FE (não é a linha de predição actual)."
            )
    else:
        notes.append(
            "Inferência servida pelo **pipeline sklearn** serializado em joblib (`model_path`). "
            "A probabilidade vem de `predict_proba` quando o estimador suporta."
        )
        if baseline_holdout:
            notes.append(
                "Compare `baseline_holdout_metrics` (sklearn simples no Baseline) com `served_holdout_metrics` "
                "(FE promovido) para ver o ganho do feature engineering."
            )

    notes.append(
        "Valores de `served_holdout_metrics` referem-se ao **conjunto de teste do treino** "
        "daquele pipeline run (FE), não ao indivíduo deste pedido."
    )

    summary_lines = [
        f"Backend: **{backend}** · modelo declarado: `{predict_model}`.",
    ]
    if sk_benchmark:
        summary_lines.append(f"Classificador de referência (CV / estudo no FE): **{sk_benchmark}**.")
    if best_cv_f is not None:
        om = m.get("optimization_metric")
        summary_lines.append(
            f"Melhor score de CV ({om or 'métrica configurada'}): **{best_cv_f:.4f}**."
        )
    if backend == "mlp" and isinstance(m.get("mlp_training"), dict):
        tr = m["mlp_training"]
        be = tr.get("best_epoch")
        if be is not None:
            summary_lines.append(f"MLP: melhor época (early stopping): **{be}**.")

    if baseline_holdout and served:
        for label, getter in (
            ("Recall", lambda b, s: (b.recall, s.recall)),
            ("F1", lambda b, s: (b.f1, s.f1)),
            ("ROC-AUC", lambda b, s: (b.roc_auc, s.roc_auc)),
        ):
            bv, sv = getter(baseline_holdout, served)
            if bv is not None and sv is not None:
                summary_lines.append(
                    f"Teste holdout — Baseline vs servido ({'MLP' if backend == 'mlp' else 'FE sklearn'}) — **{label}**: "
                    f"{bv:.4f} → {sv:.4f}."
                )
                break

    mlp_training = m.get("mlp_training") if isinstance(m.get("mlp_training"), dict) else None

    return InferenceReport(
        inference_backend=backend,
        predict_model=predict_model,
        sklearn_benchmark_classifier=sk_benchmark,
        optimization_metric=m.get("optimization_metric"),
        best_cv_score=best_cv_f,
        classification_decision_threshold=thr_f,
        served_holdout_metrics=served,
        sklearn_holdout_test=sk_holdout if backend == "mlp" else None,
        baseline_reference=baseline_ref,
        baseline_holdout_metrics=baseline_holdout,
        fe_model_comparison=fe_comparison,
        mlp_training_summary=mlp_training,
        notes=notes,
        summary_lines=summary_lines,
    )
