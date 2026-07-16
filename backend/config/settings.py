from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    databricks_host: str
    databricks_token: str
    databricks_catalog: str = "workspace"
    databricks_schema: str = "olist"
    databricks_warehouse_id: str | None = None

    groq_api_key: str | None = None
    boss_llm_model: str = "openai/gpt-oss-20b"

    mongodb_uri: str | None = None
    mongodb_db_name: str = "ai_boardroom"


settings = Settings()
