from typing import Any, Optional
from pathlib import Path
from functools import lru_cache

from pydantic import PostgresDsn, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    POSTGRES_HOST: Optional[str] = "127.0.0.1"
    POSTGRES_USER: Optional[str] = "postgres"
    POSTGRES_PASSWORD: Optional[str] = "password"
    POSTGRES_DB: Optional[str] = "agent_db"
    POSTGRES_PORT: Optional[int] = "5432"
    DATABASE_URI: Optional[str] = None

    @field_validator("DATABASE_URI", mode="before")
    def assemble_db_connection(cls, v: Optional[str], info: ValidationInfo) -> Any:
        assert info.data is not None
        values = info.data
        if isinstance(v, str):
            return v
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                host=values.get("POSTGRES_HOST"),
                path=values.get("POSTGRES_DB"),
                port=values.get("POSTGRES_PORT"),
                username=values.get("POSTGRES_USER"),
                password=values.get("POSTGRES_PASSWORD"),
            )
        )

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),  # src/cue/.env
        env_file_encoding="utf-8",
        from_attributes=True,
        extra="ignore",
        use_enum_values=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
