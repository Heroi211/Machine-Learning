from fastapi import APIRouter,HTTPException, status,Depends,Response
from schemas import roles_schemas as roles_schemas
from models.roles import Roles as roles_models
from models.users import Users as users_models
from core.deps import get_session, get_current_user
from services import roles_services as roles_service

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from typing import List


router = APIRouter()

#POST role
@router.post('/',status_code=status.HTTP_201_CREATED,response_model=roles_schemas.role)
async def post_role(role:roles_schemas.role,db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged == 3:
            new_role:roles_models = await roles_service.register_role(role,db)
            return new_role
        else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
    

#GET roles
@router.get('/',status_code=status.HTTP_200_OK,response_model=List[roles_schemas.role])
async def get_roles(db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged == 3:
            roles:List[roles_schemas.role] = await roles_service.select_all_roles(db)
            return roles
        else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")

#GET role
@router.get('/{id_role}',status_code=status.HTTP_200_OK, response_model=roles_schemas.role)
async def get_role(id_role:int, db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged == 3:
            role:roles_schemas.role = await roles_service.select_role(id_role,db)
            if role:
                return role
            else:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        else:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code == status.HTTP_400_BAD_REQUEST:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        elif e.status_code == status.HTTP_400_BAD_REQUEST:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail='Usuário não encontrado.')

#PUT role
@router.put('/{id_role}',status_code=status.HTTP_202_ACCEPTED,response_model=roles_schemas.role)
async def put_role(id_role:int, role:roles_schemas.role_update,db:AsyncSession = Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged == 3:
            role_update:roles_schemas.role = await roles_service.update_role(id_role,role,db)
            return role_update
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
    
    
    

#DELETE role
@router.delete('/{id_role}',status_code=status.HTTP_202_ACCEPTED)
async def delete_role(id_role:int,db:AsyncSession= Depends(get_session),user_logged :users_models = Depends(get_current_user)):
    try:
        if user_logged == 3:
            await roles_service.drop_role(id_role,db)
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    except HTTPException as e:
        if e.status_code != status.HTTP_401_UNAUTHORIZED:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ocorreu um erro durante a solicitação.")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Você não possui permissão para consultar esses dados.")
    
    