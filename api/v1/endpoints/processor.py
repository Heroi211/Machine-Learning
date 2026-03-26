from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from models.users import Users as users_models
from core.deps import get_session, get_current_user
from schemas import processor_schemas
from services.processor import processor_service
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/baseline",status_code=status.HTTP_201_CREATED,response_model=processor_schemas.PipelineRunResponse,)
async def run_baseline(file: UploadFile = File(...),objective: str = Form(...),db: AsyncSession = Depends(get_session),user_logged: users_models = Depends(get_current_user)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Apenas arquivos .csv são aceitos.")

    try:
        run = await processor_service.run_baseline(file=file,objective=objective,user_id=user_logged.id,db=db,)
        return run
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"Erro ao executar baseline: {str(e)}")


@router.post("/feature-engineering",status_code=status.HTTP_201_CREATED,response_model=processor_schemas.PipelineRunResponse,)
async def run_feature_engineering(file: UploadFile = File(...),objective: str = Form(...),db: AsyncSession = Depends(get_session),user_logged: users_models = Depends(get_current_user),):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Apenas arquivos .csv são aceitos.")

    try:
        run = await processor_service.run_feature_engineering(file=file,objective=objective,user_id=user_logged.id,db=db)
        return run
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"Erro ao executar feature engineering: {str(e)}")


@router.post("/predict",status_code=status.HTTP_200_OK,response_model=processor_schemas.PredictResponse)
async def predict(payload: processor_schemas.PredictRequest,db: AsyncSession = Depends(get_session),user_logged: users_models = Depends(get_current_user)):
    try:
        prediction = await processor_service.predict_single(pipeline_run_id=payload.pipeline_run_id,features=payload.features,user_id=user_logged.id,db=db)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,detail=f"Erro ao realizar predição: {str(e)}")
