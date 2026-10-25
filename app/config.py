from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""

    gemini_model: str = "gemini-3.8-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    data_dir: Path = Path("data")
    random_seed: int = 42
    corpus_size: int = 50_000


settings = Settings()