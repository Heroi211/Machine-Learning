import json
import os

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from starlette.responses import FileResponse
from models.users import Users as users_models
from core.deps import get_session, get_current_user, require_admin
from schemas import processor_schemas
from services.processor import processor_service
from services.processor.deployment_service import NoActiveDeploymentError, promote_pipeline_run
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/predict", status_code=status.HTTP_200_OK, response_model=processor_schemas.PredictResponse)
async def predict(
    payload: processor_schemas.PredictRequest,
    db: AsyncSession = Depends(get_session),
    user_logged: users_models = Depends(get_current_user),
):
    try:
        pred = await processor_service.predict_for_domain(
            domain=payload.domain,
            features=payload.features,
            user_id=user_logged.id,
            db=db,
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


@router.post(
    "/admin/promote",
    status_code=status.HTTP_201_CREATED,
    response_model=processor_schemas.DeployedModelResponse,
)
async def admin_promote(
    payload: processor_schemas.PromoteRequest,
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    try:
        dep = await promote_pipeline_run(
            domain=payload.domain,
            pipeline_run_id=payload.pipeline_run_id,
            promoted_by_user_id=admin.id,
            db=db,
        )
        return dep
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


def _file_response_for_run(run, pipeline_type: str) -> FileResponse:
    """Devolve o CSV em disco no corpo da resposta + metadados em cabeçalhos HTTP."""
    if run.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"run_id": run.id, "error": run.error_message},
        )
    if not run.csv_output_path or not os.path.isfile(run.csv_output_path):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Arquivo CSV de saída não encontrado após o pipeline.",
        )
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
)
async def admin_train_baseline(
    file: UploadFile = File(...),
    objective: str = Form(...),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    try:
        run = await processor_service.run_baseline(file=file, objective=objective, user_id=admin.id, db=db)
        return _file_response_for_run(run, "baseline")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Treino baseline falhou: {str(e)}",
        )


@router.post(
    "/admin/train/feature-engineering",
    status_code=status.HTTP_201_CREATED,
    response_class=FileResponse,
)
async def admin_train_feature_engineering(
    file: UploadFile = File(...),
    objective: str = Form(...),
    optimization_metric: str = Form("accuracy"),
    db: AsyncSession = Depends(get_session),
    admin: users_models = Depends(require_admin),
):
    try:
        run = await processor_service.run_feature_engineering(
            file=file,
            objective=objective,
            user_id=admin.id,
            db=db,
            optimization_metric=optimization_metric,
        )
        return _file_response_for_run(run, "feature_engineering")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Treino FE falhou: {str(e)}",
        )
