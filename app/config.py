from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""

    data_dir: Path = Path("data")
    random_seed: int = 42
    corpus_size: int = 50_000


settings = Settings()
