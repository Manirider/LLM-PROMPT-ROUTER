from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(...)
    openai_model: str = Field(default="gpt-4o-mini")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    log_file_path: str = Field(default="logs/route_log.jsonl")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)
    ollama_base_url: str = Field(default="http://localhost:11434/v1")
    use_ollama: bool = Field(default=True)
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


settings = Settings()
