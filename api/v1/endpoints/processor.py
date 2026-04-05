import json
import os
import uuid

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse

from core.configs import settings
from core.deps import get_current_user, get_session, require_admin
from models.users import Users as users_models
from schemas import processor_schemas
from services.processor import processor_service
from services.processor.deployment_service import NoActiveDeploymentError, RollbackError, get_deployment_history, promote_pipeline_run, rollback_deployment

router = APIRouter()

AIRFLOW_BASE_URL = settings.airflow_base_url
AIRFLOW_USER = settings.airflow_user
AIRFLOW_PASSWORD = settings.airflow_password
ML_SHARED_PATH = settings.ml_shared_path


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=processor_schemas.PredictResponse)
async def predict(payload: processor_schemas.PredictRequest, db: AsyncSession = Depends(get_session), user_logged: users_models = Depends(get_current_user)):
    try:
        pred = await processor_service.predict_for_domain(
            domain=payload.domain, features=payload.features, user_id=user_logged.id, db=db
        )
        return processor_schemas.PredictResponse(
            id=pred.id,
            domain=payload.domain.strip().lower(),
            pipeline_run_id=pred.pipeline_run_id,
            prediction=pred.prediction,
            probability=pred.probability,
            input_data=pred.input_data if isinstance(pred.input_data, dict) else dict(pred.input_data),
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
async def admin_promote(payload: processor_schemas.PromoteRequest, db: AsyncSession = Depends(get_session), admin: users_models = Depends(require_admin)):
    try:
        dep = await promote_pipeline_run(
            domain=payload.domain, pipeline_run_id=payload.pipeline_run_id, promoted_by_user_id=admin.id, db=db
        )
        return dep
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/admin/train/trigger-dag", status_code=status.HTTP_202_ACCEPTED, response_model=processor_schemas.TriggerDagResponse)
async def admin_trigger_dag(
    file: UploadFile = File(...), objective: str = Form(...), optimization_metric: str = Form("accuracy"),
    time_limit_minutes: int = Form(2), acc_target: float = Form(0.90), admin: users_models = Depends(require_admin),
):
    """Grava o CSV em volume compartilhado e dispara o DAG de treino no Airflow."""
    upload_dir = os.path.join(ML_SHARED_PATH)
    os.makedirs(upload_dir, exist_ok=True)
    filename = f"{objective}_{uuid.uuid4().hex[:8]}_{file.filename}"
    csv_path = os.path.join(upload_dir, filename)
    content = await file.read()
    with open(csv_path, "wb") as f:
        f.write(content)

    dag_run_id = f"manual__{objective}_{uuid.uuid4().hex[:8]}"
    dag_conf = {
        "objective": objective,
        "csv_path": csv_path,
        "optimization_metric": optimization_metric,
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
        objective=objective,
        csv_path=csv_path,
        message=f"DAG disparado. Acompanhe em {AIRFLOW_BASE_URL}/dags/ml_training_pipeline/grid",
    )


@router.get("/admin/deployments/{domain}/history", status_code=status.HTTP_200_OK, response_model=list[processor_schemas.DeployedModelResponse])
async def admin_deployment_history(domain: str, db: AsyncSession = Depends(get_session), admin: users_models = Depends(require_admin)):
    """Lista os últimos deployments (active + archived) do domínio, do mais recente ao mais antigo."""
    records = await get_deployment_history(domain=domain, db=db)
    if not records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Nenhum deployment encontrado para o domínio '{domain}'."
        )
    return records


@router.post("/admin/rollback", status_code=status.HTTP_200_OK, response_model=processor_schemas.DeployedModelResponse)
async def admin_rollback(payload: processor_schemas.RollbackRequest, db: AsyncSession = Depends(get_session), admin: users_models = Depends(require_admin)):
    """Reverte para o deployment archived mais recente do domínio, arquivando o active atual."""
    try:
        dep = await rollback_deployment(domain=payload.domain, db=db)
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


@router.post("/admin/train/baseline", status_code=status.HTTP_201_CREATED, response_class=FileResponse)
async def admin_train_baseline(
    file: UploadFile = File(...), objective: str = Form(...), db: AsyncSession = Depends(get_session), admin: users_models = Depends(require_admin)
):
    try:
        run = await processor_service.run_baseline(file=file, objective=objective, user_id=admin.id, db=db)
        return _file_response_for_run(run, "baseline")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Treino baseline falhou: {str(e)}")


@router.post("/admin/train/feature-engineering", status_code=status.HTTP_201_CREATED, response_class=FileResponse)
async def admin_train_feature_engineering(
    file: UploadFile = File(...), objective: str = Form(...), optimization_metric: str = Form("accuracy"),
    time_limit_minutes: int = Form(2), acc_target: float = Form(0.90), db: AsyncSession = Depends(get_session), admin: users_models = Depends(require_admin),
):
    try:
        run = await processor_service.run_feature_engineering(
            file=file, objective=objective, user_id=admin.id, db=db,
            optimization_metric=optimization_metric, time_limit_minutes=time_limit_minutes, acc_target=acc_target,
        )
        return _file_response_for_run(run, "feature_engineering")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Treino FE falhou: {str(e)}")
