import json
import os
import uuid
from typing import Literal

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
from services.processor import processor_service
from services.processor.deployment_service import (
    NoActiveDeploymentError,
    RollbackError,
    get_deployment_history,
    promote_active_feature_engineering_for_objective,
    rollback_deployment,
)
from services.processor.pipeline_runs_service import list_pipeline_runs

router = APIRouter()

AIRFLOW_BASE_URL = settings.airflow_base_url
AIRFLOW_USER = settings.airflow_user
AIRFLOW_PASSWORD = settings.airflow_password
ML_SHARED_PATH = settings.ml_shared_path


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=processor_schemas.PredictResponse)
async def predict(payload: processor_schemas.PredictRequest, db: AsyncSession = Depends(get_session), user_logged: users_models = Depends(get_current_user)):
    try:
        features_dict = payload.features.model_dump(mode="json", by_alias=True)
        pred, inference_report = await processor_service.predict_for_domain(
            domain=payload.domain, features=features_dict, user_id=user_logged.id, db=db
        )
        prob_pct = None
        if pred.probability is not None:
            prob_pct = round(float(pred.probability) * 100, 2)
        return processor_schemas.PredictResponse(
            id=pred.id,
            domain=payload.domain,
            pipeline_run_id=pred.pipeline_run_id,
            prediction=pred.prediction,
            probability=prob_pct,
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
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """
    Promove para inferência em ``/predict`` o **único** run de feature engineering activo e concluído,
    com ``objective`` igual a **OBJECTIVE** (env), à imagem das rotas síncronas de baseline/FE.
    """
    try:
        objective = settings.objective.strip().lower()
        dep = await promote_active_feature_engineering_for_objective(
            objective=objective,
            promoted_by_user_id=admin.id,
            db=db,
        )
        return dep
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/admin/train/trigger-dag", status_code=status.HTTP_202_ACCEPTED, response_model=processor_schemas.TriggerDagResponse)
async def admin_trigger_dag(
    file: UploadFile = File(...),
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Form("accuracy"),
    min_precision: float | None = Form(None, description="Guardrail opcional: precisão mínima [0,1]."),
    min_roc_auc: float | None = Form(None, description="Guardrail opcional: ROC-AUC mínimo [0,1]."),
    tuning_n_iter: int | None = Form(None, description="Número máximo de amostras no tuning (opcional)."),
    time_limit_minutes: int = Form(2),
    acc_target: float | None = Form(None),
    admin: users_models = Depends(require_airflow_api_trigger_enabled),
):
    """Grava o CSV em volume partilhado e dispara o DAG; **objective** vem de OBJECTIVE na env."""
    obj = settings.objective.strip().lower()
    upload_dir = os.path.join(ML_SHARED_PATH)
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{obj}_{uuid.uuid4().hex[:8]}_{file.filename}"
    csv_path = os.path.join(upload_dir, filename)
    content = await file.read()
    with open(csv_path, "wb") as f:
        f.write(content)

    dag_run_id = f"manual__{obj}_{uuid.uuid4().hex[:8]}"
    dag_conf = {
        "objective": obj,
        "csv_path": csv_path,
        "optimization_metric": optimization_metric,
        "min_precision": min_precision,
        "min_roc_auc": min_roc_auc,
        "tuning_n_iter": tuning_n_iter,
        "time_limit_minutes": time_limit_minutes,
        "acc_target": acc_target,
        "user_id": admin.id,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{AIRFLOW_BASE_URL}/api/v1/dags/ml_training_pipeline/dagRuns",
                json={"dag_run_id": dag_run_id, "conf": dag_conf},
                auth=(AIRFLOW_USER, AIRFLOW_PASSWORD),
            )
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Airflow recusou o trigger: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Airflow indisponível: {str(e)}")

    return processor_schemas.TriggerDagResponse(
        dag_run_id=dag_run_id,
        dag_id="ml_training_pipeline",
        objective=obj,
        csv_path=csv_path,
        message=f"DAG disparado. Acompanhe em {AIRFLOW_BASE_URL}/dags/ml_training_pipeline/grid",
    )


@router.get("/admin/runs", status_code=status.HTTP_200_OK, response_model=list[processor_schemas.PipelineRunResponse])
async def admin_list_pipeline_runs(
    pipeline_type: Literal["baseline", "feature_engineering"] | None = Query(None, description="Tipo de pipeline."),
    run_status: Literal["processing", "completed", "failed"] | None = Query(
        None, alias="status", description="Estado da execução."
    ),
    limit: int = Query(50, ge=1, le=200, description="Máximo de registos devolvidos (mais recentes primeiro)."),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Lista runs de pipeline para **OBJECTIVE** (env); `pipeline_type` e `status` continuam opcionais na query."""
    objective = settings.objective.strip().lower()
    return await list_pipeline_runs(
        db,
        objective=objective,
        pipeline_type=pipeline_type,
        status=run_status,
        limit=limit,
    )


@router.get("/admin/deployments/history", status_code=status.HTTP_200_OK, response_model=list[processor_schemas.DeployedModelResponse])
async def admin_deployment_history(
    db: AsyncSession = Depends(get_session), admin: users_models = Depends(require_admin)
):
    """Lista os últimos deployments (**OBJECTIVE** na env), do mais recente ao mais antigo."""
    domain = settings.objective.strip().lower()
    records = await get_deployment_history(domain=domain, db=db)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum deployment encontrado para o objective '{domain}' (OBJECTIVE na env).",
        )
    return records


@router.post("/admin/rollback", status_code=status.HTTP_200_OK, response_model=processor_schemas.DeployedModelResponse)
async def admin_rollback(
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    """Reverte para o deployment archived mais recente (**OBJECTIVE** na env), arquivando o active actual."""
    try:
        domain = settings.objective.strip().lower()
        dep = await rollback_deployment(domain=domain, db=db)
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
            "X-Pipeline-Metrics": json.dumps(run.metrics, ensure_ascii=False) if run.metrics else "",
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
    optimization_metric: Literal["accuracy", "precision", "recall", "f1", "roc_auc"] = Form("accuracy"),
    min_precision: float | None = Form(None, description="Guardrail opcional: precisão mínima [0,1]."),
    min_roc_auc: float | None = Form(None, description="Guardrail opcional: ROC-AUC mínimo [0,1]."),
    tuning_n_iter: int | None = Form(None, description="Número máximo de amostras no tuning (opcional)."),
    time_limit_minutes: int = Form(2),
    acc_target: float | None = Form(None),
    decision_threshold: float | None = Form(
        None,
        description="P(classe positiva) mínima para métricas de teste; omite = env CLASSIFICATION_DECISION_THRESHOLD.",
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
                "X-Pipeline-Metrics": json.dumps(run.metrics, ensure_ascii=False) if run.metrics else "",
            },
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Treino FE falhou: {str(e)}")
