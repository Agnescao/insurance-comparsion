from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Insurance Comparison Prototype"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    data_output_dir: Path = data_dir / "output"
    backend_data_dir: Path = project_root / "backend" / "data"
    sqlite_path: Path = backend_data_dir / "app.sqlite3"
    dump_parse_output: bool = True

    # Embedding settings (online provider is optional; local hash fallback is always available)
    embedding_provider: str = Field(default="hash", description="hash or qwen")
    embedding_dim: int = 256
    qwen_api_key: str | None = None
    qwen_embedding_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    qwen_embedding_model: str = "text-embedding-v3"

    # Milvus settings (optional)
    milvus_enabled: bool = False
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_user: str | None = None
    milvus_password: str | None = None
    milvus_db_name: str = "default"

    chunk_size: int = 900
    chunk_overlap: int = 120
    semantic_merge_threshold: float = 0.86

    cors_allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    cors_allow_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
settings.backend_data_dir.mkdir(parents=True, exist_ok=True)
settings.data_output_dir.mkdir(parents=True, exist_ok=True)
