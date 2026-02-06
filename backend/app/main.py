import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.v1.router import router as v1_router

# 로깅 초기화
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ImputeX API",
    description="Missing value imputation API",
    version="1.0.0",
)

# CORS middleware
origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(v1_router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def on_startup():
    logger.info("ImputeX API started", extra={"version": "1.0.0"})


@app.get("/health")
async def health_check():
    from app.db.session import engine
    try:
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {"status": "healthy", "version": "1.0.0", "database": db_status}
