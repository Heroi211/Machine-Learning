import json
import logging
import math
import os
import uuid
from datetime import date, datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from core.configs import settings
from core.deps import (
    get_current_user,
    get_session,
    require_admin,
    require_airflow_api_trigger_enabled,
    require_sync_training_routes_enabled,
)
from models.users import Users as users_models
from schemas import processor_schemas
from ml_core_ring.paths import airflow_upload_path, resolved_ml_shared_uploads_dir
from platform_ring.promote_response import build_deployed_model_response
from platform_ring.promote_service import promote_for_domain
from platform_ring.runs_service import list_runs_for_domain
from platform_ring.training_trigger import trigger_training_dag
from services.processor import processor_service
from services.processor.deployment_service import (
    NoActiveDeploymentError,
    RollbackError,
    get_deployment_history,
    rollback_deployment,
)
from services.processor.pipeline_run_view import build_pipeline_run_view

router = APIRouter()
_logger_processor_ep = logging.getLogger(__name__)


def _metrics_json_for_response_header(metrics: dict | None) -> str:
    """Serializa métricas para o cabeçalho ``X-Pipeline-Metrics``.

    O Starlette codifica valores de cabeçalho em **latin-1**. JSON com ``ensure_ascii=False``
    pode incluir caracteres > U+00FF; ao fazer ``encode('latin-1')`` levanta-se
    ``UnicodeEncodeError``, que em Python é **subclasse de ValueError** — o endpoint
    capturava isso e devolvia HTTP 400 sem relação com validação de formulário.
    Por isso usamos ``ensure_ascii=True`` (sequências ``\\uXXXX`` só com ASCII).
    """

    def norm(o: Any) -> Any:
        if o is None:
            return None
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [norm(v) for v in o]
        if isinstance(o, bool):
            return o
        if isinstance(o, float):
            return None if not math.isfinite(o) else o
        if isinstance(o, int):
            return o
        if isinstance(o, str):
            return o
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if hasattr(o, "tolist") and callable(o.tolist):
            try:
                return norm(o.tolist())
            except Exception:
                pass
        if hasattr(o, "item") and callable(o.item):
            try:
                return norm(o.item())
            except Exception:
                pass
        return str(o)

    if not metrics:
        return ""
    try:
        payload = norm(metrics)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as e:
        _logger_processor_ep.warning("X-Pipeline-Metrics: fallback após falha de serialização: %s", e)
        return "{}"


ML_SHARED_PATH = resolved_ml_shared_uploads_dir()


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=processor_schemas.PredictResponse)
async def predict(payload: processor_schemas.PredictRequest, db: AsyncSession = Depends(get_session), user_logged: users_models = Depends(get_current_user)):
    try:
        features_dict = payload.features.model_dump(mode="json", by_alias=True)
        pred, inference_report, reco_extra = await processor_service.predict_for_domain(
            domain=payload.domain, features=features_dict, user_id=user_logged.id, db=db
        )
        prob_pct = None
        if pred.probability is not None:
            prob_pct = round(float(pred.probability) * 100, 2)
        prob_display = f"{prob_pct}%" if prob_pct is not None else None
        recommended = reco_extra.get("recommended_items") if reco_extra else None
        return processor_schemas.PredictResponse(
            id=pred.id,
            domain=payload.domain,
            pipeline_run_id=pred.pipeline_run_id,
            prediction=pred.prediction,
            probability=prob_pct,
            probability_display=prob_display,
            recommended_items=recommended,
            input_data=pred.input_data if isinstance(pred.input_data, dict) else dict(pred.input_data),
            inference_report=inference_report,
        )
    except NoActiveDeploymentError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao realizar predição: {str(e)}",
        )


@router.post("/admin/promote", status_code=status.HTTP_201_CREATED, response_model=processor_schemas.DeployedModelResponse)
async def admin_promote(
    domain: str | None = Query(
        None,
        description=(
            "Domínio a promover. Obrigatório na prática para recommendation "
            "(default OBJECTIVE na env é churn). Ex.: recommendation, churn."
        ),
    ),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Promove o run activo do domínio para servir em ``/predict`` (tabular FE ou recomendação)."""
    try:
        objective = (domain or settings.objective).strip().lower()
        result = await promote_for_domain(
            domain=objective,
            promoted_by_user_id=admin.id,
            db=db,
        )
        return build_deployed_model_response(result.deployment, result.mlflow_registry)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/admin/train/trigger-dag", status_code=status.HTTP_202_ACCEPTED, response_model=processor_schemas.TriggerDagResponse)
async def admin_trigger_dag(
    domain: str | None = Form(
        None,
        description="Domínio ML (churn, recommendation, …). Default: OBJECTIVE na env.",
    ),
    file: UploadFile | None = File(
        None,
        description="CSV obrigatório para domínios tabulares; omitir para recommendation.",
    ),
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Form("accuracy"),
    min_precision: float | None = Form(None, description="Guardrail opcional: precisão mínima [0,1]."),
    min_roc_auc: float | None = Form(None, description="Guardrail opcional: ROC-AUC mínimo [0,1]."),
    tuning_n_iter: int | None = Form(None, description="Número máximo de amostras no tuning (opcional)."),
    time_limit_minutes: int = Form(2),
    acc_target: float | None = Form(None),
    top_k: int | None = Form(None, description="Top-K para domínio recommendation."),
    admin: users_models = Depends(require_airflow_api_trigger_enabled),
):
    """Dispara ``ml_training_dispatch`` no Airflow (treino por ``domain``)."""
    obj = (domain or settings.objective).strip().lower()
    csv_path: str | None = None

    if file is not None:
        upload_dir = ML_SHARED_PATH
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"{obj}_{uuid.uuid4().hex[:8]}_{file.filename}"
        host_path = os.path.join(upload_dir, filename)
        content = await file.read()
        with open(host_path, "wb") as f:
            f.write(content)
        csv_path = airflow_upload_path(filename)

    extra: dict[str, Any] = {}
    if top_k is not None:
        extra["top_k"] = top_k

    try:
        result = await trigger_training_dag(
            domain=obj,
            user_id=admin.id,
            csv_path=csv_path,
            optimization_metric=optimization_metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            time_limit_minutes=time_limit_minutes,
            acc_target=acc_target,
            extra=extra or None,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Airflow recusou o trigger: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Airflow indisponível: {str(e)}")

    return processor_schemas.TriggerDagResponse(
        dag_run_id=result.dag_run_id,
        dag_id=result.dag_id,
        domain=result.domain,
        objective=result.domain,
        csv_path=result.csv_path,
        message=result.message,
    )


@router.get(
    "/admin/runs",
    status_code=status.HTTP_200_OK,
    summary="Listar runs (formato estruturado)",
    description=(
        "Devolve runs reorganizados em secções nomeadas: `run`, `training_context`, "
        "`inference`, `experiments`, `comparison`, `baseline_reference_snapshot` e "
        "`_legacy_flat_metrics` (cópia integral do antigo blob `metrics` para "
        "compatibilidade com clientes antigos)."
    ),
)
async def admin_list_pipeline_runs(
    domain: str | None = Query(
        None,
        description=(
            "Filtrar por domínio. Para runs de recomendação use recommendation "
            "(default OBJECTIVE na env é churn — omitir devolve runs churn/vazio)."
        ),
    ),
    pipeline_type: Literal["baseline", "feature_engineering", "recommendation"] | None = Query(
        None, description="Tipo de pipeline."
    ),
    run_status: Literal["processing", "completed", "failed"] | None = Query(
        None, alias="status", description="Estado da execução."
    ),
    limit: int = Query(50, ge=1, le=200, description="Máximo de registos devolvidos (mais recentes primeiro)."),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Lista runs de pipeline por domínio."""
    objective = (domain or settings.objective).strip().lower()
    runs = await list_runs_for_domain(
        db,
        domain=objective,
        pipeline_type=pipeline_type,
        status=run_status,
        limit=limit,
    )
    return [build_pipeline_run_view(r) for r in runs]


@router.get("/admin/deployments/history", status_code=status.HTTP_200_OK, response_model=list[processor_schemas.DeployedModelResponse])
async def admin_deployment_history(
    domain: str | None = Query(None, description="Domínio (default: OBJECTIVE na env)."),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Lista os últimos deployments do domínio, do mais recente ao mais antigo."""
    resolved = (domain or settings.objective).strip().lower()
    records = await get_deployment_history(domain=resolved, db=db)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum deployment encontrado para o domínio '{resolved}' (OBJECTIVE na env).",
        )
    return records


@router.post("/admin/rollback", status_code=status.HTTP_200_OK, response_model=processor_schemas.DeployedModelResponse)
async def admin_rollback(
    domain: str | None = Query(None, description="Domínio (default: OBJECTIVE na env)."),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Reverte para o deployment archived mais recente do domínio."""
    try:
        resolved = (domain or settings.objective).strip().lower()
        dep = await rollback_deployment(domain=resolved, db=db)
        return dep
    except RollbackError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Rollback falhou: {str(e)}")


def _file_response_for_run(run, pipeline_type: str) -> FileResponse:
    """Devolve o CSV em disco no corpo da resposta + metadados em cabeçalhos HTTP."""
    if run.status != "completed":
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"run_id": run.id, "error": run.error_message})
    if not run.csv_output_path or not os.path.isfile(run.csv_output_path):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Arquivo CSV de saída não encontrado após o pipeline.")
    return FileResponse(
        path=run.csv_output_path,
        filename=os.path.basename(run.csv_output_path),
        media_type="text/csv",
        status_code=status.HTTP_201_CREATED,
        headers={
            "X-Pipeline-Run-Id": str(run.id),
            "X-Pipeline-Type": pipeline_type,
            "X-Pipeline-Objective": run.objective,
            "X-Pipeline-Metrics": _metrics_json_for_response_header(run.metrics),
        },
    )


@router.post(
    "/admin/train/baseline",
    status_code=status.HTTP_201_CREATED,
    response_class=FileResponse,
    summary="Treino baseline (domínio fixo)",
    description=("Baseline churn, arquivo tratado conforme contrato deve ser enviado."),
)
async def admin_train_baseline(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_sync_training_routes_enabled),
):
    try:
        objective = settings.objective.strip().lower()
        run = await processor_service.run_baseline(file=file, objective=objective, user_id=admin.id, db=db)
        return _file_response_for_run(run, "baseline")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Treino baseline falhou: {str(e)}")


def _schedule_remove(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


@router.post(
    "/admin/train/feature-engineering",
    status_code=status.HTTP_201_CREATED,
    response_class=FileResponse,
    summary="Treino feature-engineering (domínio fixo)",
    description=(
        "Domínio pela env **OBJECTIVE**. Manifest do baseline via ``pre_processed`` ou BD "
        "(``output_sample_csv_stable`` no manifest). Nesta versão **não** há upload de CSV na rota."
    ),
)
async def admin_train_feature_engineering(
    background_tasks: BackgroundTasks,
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Form("recall"),
    min_precision: float | None = Form(None, description="Guardrail opcional: precisão mínima [0,1]."),
    min_roc_auc: float | None = Form(None, description="Guardrail opcional: ROC-AUC mínimo [0,1]."),
    tuning_n_iter: int | None = Form(None, description="Número máximo de amostras no tuning (opcional)."),
    time_limit_minutes: int = Form(2),
    acc_target: float | None = Form(None),
    decision_threshold: float = Form(
        0.3,
        description="P(classe positiva) mínima para métricas de teste; padrão 0,3.",
    ),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_sync_training_routes_enabled),
):
    try:
        objective = settings.objective.strip().lower()
        run, zip_path = await processor_service.run_feature_engineering(
            objective=objective, user_id=admin.id, db=db,
            optimization_metric=optimization_metric,
            min_precision=min_precision,
            min_roc_auc=min_roc_auc,
            tuning_n_iter=tuning_n_iter,
            time_limit_minutes=time_limit_minutes,
            acc_target=acc_target,
            decision_threshold=decision_threshold,
        )
        if run.status != "completed":
            if zip_path:
                _schedule_remove(zip_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"run_id": run.id, "error": run.error_message or "Pipeline não concluiu com sucesso."},
            )
        if not zip_path or not os.path.isfile(zip_path):
            if zip_path:
                _schedule_remove(zip_path)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Arquivo ZIP de artefatos não foi gerado após o pipeline.",
            )
        background_tasks.add_task(_schedule_remove, zip_path)
        return FileResponse(
            path=zip_path,
            filename=f"fe_artifacts_run_{run.id}.zip",
            media_type="application/zip",
            status_code=status.HTTP_201_CREATED,
            headers={
                "X-Pipeline-Run-Id": str(run.id),
                "X-Pipeline-Type": "feature_engineering",
                "X-Pipeline-Objective": run.objective,
                "X-Pipeline-Metrics": _metrics_json_for_response_header(run.metrics),
            },
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Treino FE falhou: {str(e)}")
