"""Fornece operacoes de autenticacao usadas pelos endpoints da API."""

from models.users import Users as users_models
from schemas import users_schemas as users_schemas
from sqlalchemy.ext.asyncio import AsyncSession
from core.security import get_password_hash
from core.auth import authenticate_user


async def authorize(email: str, password: str, db: AsyncSession):
    """Retorna o usuario autenticado quando as credenciais sao validas."""
    user = await authenticate_user(email, password, db)
    return user


async def register_user(user: users_schemas.users_create, db: AsyncSession) -> users_models:
    """Cria e persiste um usuario com a senha em formato de hash."""
    new_user: users_models = users_models(name=user.name, email=user.email,
                                          password=get_password_hash(user.password))
    async with db as session:
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        return new_user
