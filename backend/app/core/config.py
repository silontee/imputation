from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_ENV: str = "dev"
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = "postgresql+psycopg2://imputex:imputex_password@localhost:5432/imputex"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # File storage (local, NAS 연동 시 마운트 경로 변경)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    RESULT_DIR: Path = BASE_DIR / "results"

    # CORS
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Upload limits
    MAX_UPLOAD_MB: int = 500
    SAMPLE_ROWS: int = 1000

    # TOTEM model paths
    TOTEM_MODEL_PATH: str = "/app/models/totem"
    TOTEM_CONFIG_PATH: str = ""  # Empty string means auto-discover from MODEL_PATH
    TOTEM_CODE_PATH: str = "/app/TOTEM/imputation"


settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)
