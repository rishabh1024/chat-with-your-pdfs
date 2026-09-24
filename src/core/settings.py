from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]


class MongoDBSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    user: SecretStr = Field(alias="MONGODB_USER")
    password: SecretStr = Field(alias="MONGODB_PASSWORD")
    cluster_id: str = Field(alias="MONGODB_CLUSTER_ID")
    db_name: str = Field(default="sample_mflix", alias="MONGO_DB_NAME")


class SupabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        url_preserve_empty_path=True,
    )

    key: SecretStr = Field(alias="SUPABASE_KEY")
    url: HttpUrl = Field(alias="SUPABASE_URL")
    auth_url: HttpUrl = Field(alias="SUPABASE_AUTH_URL")
    db_url: str = Field(alias="SUPABASE_DB_URL")


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    openrouter_api_key: SecretStr = Field(alias="OPENROUTER_API_KEY")


class LangsmithSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    api_key: SecretStr | None = Field(default=None, alias="LANGSMITH_API_KEY")
    project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    level: LogLevel = Field(default="DEBUG", validation_alias="LOG_LEVEL")
    log_format: LogFormat = Field(default="text", validation_alias="LOG_FORMAT")


class CORSSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        alias="CORS_ALLOWED_ORIGINS",
    )
    allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")
    allow_methods: Annotated[list[str], NoDecode] = Field(default=["*"], alias="CORS_ALLOW_METHODS")
    allow_headers: Annotated[list[str], NoDecode] = Field(default=["*"], alias="CORS_ALLOW_HEADERS")

    @field_validator("allowed_origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def split_comma_separated(cls, value: str | list[str]) -> list[str]:
        """Parse comma-separated string into list, or pass through if already a list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    mongo: MongoDBSettings = MongoDBSettings()  # type: ignore[call-arg]
    supabase: SupabaseSettings = SupabaseSettings()  # type: ignore[call-arg]
    openrouter: OpenRouterSettings = OpenRouterSettings()  # type: ignore[call-arg]
    langsmith: LangsmithSettings = LangsmithSettings()  # type: ignore[call-arg]
    database: DatabaseSettings = DatabaseSettings()  # type: ignore[call-arg]
    logger: LoggingSettings = LoggingSettings()
    cors: CORSSettings = CORSSettings()  # type: ignore[call-arg]


@lru_cache
def load_environment_variables() -> AppSettings:
    return AppSettings()
