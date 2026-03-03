from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_CONFIG_PATH = Path(__file__).resolve()
_BACKEND_ROOT = _CONFIG_PATH.parents[1]
_PROJECT_ROOT = _CONFIG_PATH.parents[2]
_ENV_FILES = (_BACKEND_ROOT / ".env", _PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    app_name: str = "Insurance Comparison Prototype"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    project_root: Path = _PROJECT_ROOT
    data_dir: Path = project_root / "data"
    data_output_dir: Path = data_dir / "output"
    backend_data_dir: Path = project_root / "backend" / "data"
    sqlite_path: Path = backend_data_dir / "app.sqlite3"
    dump_parse_output: bool = True

    # Embedding settings (online provider is optional; local hash fallback is always available)
    embedding_provider: str = Field(default="hash", description="hash or qwen")
    embedding_dim: int = 1024
    qwen_api_key: str | None = None
    qwen_embedding_url: str = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    qwen_embedding_model: str = "text-embedding-v3"
    qwen_timeout_sec: int = 30
    qwen_max_retries: int = 4
    qwen_retry_backoff_sec: float = 1.2
    qwen_batch_size: int = 8
    qwen_trust_env_proxy: bool = False

    # LLM planner / answer (DashScope OpenAI-compatible endpoint)
    llm_enabled: bool = True
    llm_provider: str = "dashscope"
    llm_api_key: str | None = None
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_planner_model: str = "deepseek-v3"
    llm_answer_model: str = "deepseek-v3"
    llm_keyword_model: str = "qwen-turbo"
    llm_timeout_sec: int = 12

    # Fact extraction pipeline
    fact_extractor_mode: str = "llm"  # llm | hybrid | rule
    fact_extractor_model: str = "qwen-plus"
    fact_extractor_api_key: str | None = None
    dashscope_api_key: str | None = None
    fact_extractor_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    fact_extractor_timeout_sec: int = 45
    fact_extractor_max_tokens: int = 1200
    ingest_parallel_workers: int = 4

    # Milvus settings (optional)
    milvus_enabled: bool = True
    milvus_uri: str = "tcp://121.41.85.215:19530"
    milvus_token: str | None = None
    milvus_user: str | None = None
    milvus_password: str | None = None
    milvus_db_name: str = "default"
    milvus_chunk_hybrid_collection: str = "policy_chunks_hybrid"
    milvus_fact_hybrid_collection: str = "policy_facts_hybrid"

    chunk_size: int = 900
    chunk_overlap: int = 120
    semantic_merge_threshold: float = 0.86
    normalize_to_simplified: bool = True

    # Hybrid retrieval weights
    retrieval_dense_weight: float = 0.10
    retrieval_sparse_weight: float = 0.90

    # Compare pipeline performance
    compare_parallel_enabled: bool = True
    compare_parallel_workers: int = 4

    cors_allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    cors_allow_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    model_config = SettingsConfigDict(
        env_file=tuple(str(p) for p in _ENV_FILES),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
settings.backend_data_dir.mkdir(parents=True, exist_ok=True)
settings.data_output_dir.mkdir(parents=True, exist_ok=True)
