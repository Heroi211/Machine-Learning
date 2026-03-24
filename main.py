from fastapi import FastAPI
from core.configs import settings
from api.v1 import api
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=settings.project_name,version=settings.project_version)
app.include_router(api.router,prefix=settings.project_version)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)