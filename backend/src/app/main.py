import logging

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .api.main import api_router
from .core.config import get_settings

logger = logging.getLogger(__name__)

app = FastAPI()

# Set all CORS enabled origins
origins = ["*"]
settings = get_settings()
if settings.BACKEND_CORS_ORIGINS:
    origins = [str(origin) for origin in settings.BACKEND_CORS_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}
