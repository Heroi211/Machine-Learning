from fastapi import APIRouter,Depends
from core.deps import get_current_user
from api.v1.endpoints import users
from api.v1.endpoints import roles

router = APIRouter(dependencies=[Depends(get_current_user)])

router.include_router(router=users.router,prefix="/users",tags=["users"])
router.include_router(router=roles.router,prefix="/roles",tags=["roles"])
