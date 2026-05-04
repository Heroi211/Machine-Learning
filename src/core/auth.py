"""Funções de autenticação, validação de senha e geração de tokens JWT."""

from core.configs import settings
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from models.users import Users as users_models
from sqlalchemy.future import select
from core.security import verify_password
from datetime import datetime,timedelta
import os
from jose import jwt
from pytz import timezone as TZ

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.project_version}/auth/authenticate")

async def authenticate_user(email:str,password:str, db:AsyncSession) -> Optional[users_models]:
    """Valida email e senha, retornando o usuário autenticado quando válido."""

    async with db as session:
        querie = select(users_models).filter(users_models.email == email)
        resultset = await session.execute(querie)
        user:users_models = resultset.scalars().unique().one_or_none()

        if not user:
            return None

        if not verify_password(password,user.password):
            return None

        return user

def _generate_token(type_token:str,life_time:timedelta,sub:str)->str:
    """Gera token JWT assinado para o tipo e tempo de vida informados."""
    payload={}

    timezone = TZ(os.getenv('TIMEZONE'))
    expire = datetime.now(tz=timezone)+life_time

    payload["type"] = type_token
    payload["exp"] = expire
    payload["iat"] = datetime.now(tz=timezone)
    payload["sub"] = str(sub)

    return jwt.encode(payload,settings.jwt_secret,algorithm=settings.algorithm)

def _generate_access_token(sub:str) ->str:
    """Gera token de acesso JWT para o usuário informado."""
    return _generate_token(
        type_token='access_token',
        life_time=timedelta(minutes=float(settings.access_token_expire_minutes)),
        sub=sub
    )
