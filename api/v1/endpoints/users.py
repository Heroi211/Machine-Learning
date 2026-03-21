from fastapi import APIRouter,HTTPException,status,Depends,Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from models.users import Users as users_models

from schemas import users_schemas as users_schemas
from core.deps import get_session,get_current_user
from typing import List
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
import datetime
from services.user import users_services as users_service
from sqlalchemy.exc import IntegrityError
from core.auth import _generate_access_token
from core.security import get_password_hash

router = APIRouter()

#POST user / signup
@router.post('/signup', response_model=users_schemas.users,status_code=status.HTTP_201_CREATED)
async def post_user(user: users_schemas.users_create,db:AsyncSession = Depends(get_session)):
    try:
        new_user:users_models = await users_service.register_user(user,db)
        return new_user
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE,detail="Usuário já cadastrado na base de dados")

#POST Login
@router.post('/authorize',status_code=status.HTTP_200_OK)
async def login(form_data:OAuth2PasswordRequestForm = Depends(),db:AsyncSession = Depends(get_session)):
    user = await users_service.login_user(form_data.username,form_data.password,db)
    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Dados incorretos")
    
    return JSONResponse(content={"access_token":_generate_access_token(sub=user.id), "token_type":"bearer"},status_code=status.HTTP_200_OK)

#GET Logged
@router.get('/logged', response_model=users_schemas.users,status_code=status.HTTP_200_OK)
async def get_logged(user_logged :users_models = Depends(get_current_user)):
    return user_logged

#GET users
@router.get('/', response_model=List[users_schemas.usersGetData],status_code=status.HTTP_202_ACCEPTED)
async def get_users(db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged:
            users:List[users_schemas.usersGetData] = await users_service.select_all_users(db)
            return users
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
    
#GET user
@router.get('/{id_user}',response_model=users_schemas.users,status_code=status.HTTP_202_ACCEPTED)
async def get_user(id_user : int, db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged:
            user:users_schemas.users = await users_service.select_user(id_user,db)
            if user:
                return user
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuário não encontrado.")
        elif e.status_code == status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Ocorreu um erro durante a solicitação.")
#PUT user
@router.put('/{id_user}',status_code=status.HTTP_202_ACCEPTED)
async def put_user(id_user:int,payload:users_schemas.users_update, db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)): 
    try:
        if user_logged:
            user_id = id_user
            user_data = payload.model_dump(exclude_unset=True)
            users_schemas.users = await users_service.update_user(user_id,user_data,db)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
    
#DELETE user
@router.delete('/{id_user}',status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(id_user,db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged:
            await users_service.drop_user(id_user,db)
            return Response (status_code=status.HTTP_204_NO_CONTENT)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")

#Forgot Password - Send Email
@router.post('/forgot-password/{email}',status_code=status.HTTP_200_OK)
async def forgot_password(email:str,db:AsyncSession = Depends(get_session)):
    try:
        token = await users_service.generate_reset_token(email,db)
        await users_service.send_email(email,token)
        return {"message":"Email de redefinição de senha enviado!."}
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
        
#Reset Password
@router.post('/reset-password',status_code=status.HTTP_200_OK)
async def reset_password(token: str, password: str, db: AsyncSession = Depends(get_session)):
    try:
        user:users_schemas.users_update = await users_service.get_user_by_reset_token(token, db)
        if not user or user.reset_password_expires < datetime.datetime.now():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token inválido ou expirado")
        
        user.password=get_password_hash(password)
        user.reset_password_token = None
        user.reset_password_expires = None
        db.add(user)
        await db.commit()
        return {"message": "Senha redefinida com sucesso!"}
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Você não possui permissão para consultar esses dados.")