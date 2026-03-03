from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# Fixed defaults for one-click run.
# You can still override them via CLI when needed.
DEFAULT_MILVUS_URI = "tcp://121.41.85.215:19530"
DEFAULT_MILVUS_USER = "root"
DEFAULT_MILVUS_PASSWORD = "Milvus"
DEFAULT_MILVUS_DB_NAME = "default"
DEFAULT_EMBEDDING_PROVIDER = "qwen"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_RECREATE = True
DEFAULT_QWEN_MODEL = "text-embedding-v3"
DEFAULT_QWEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDFs and directly ingest chunks/facts into Milvus hybrid.")
    parser.add_argument("--uri", default=DEFAULT_MILVUS_URI, help="Milvus URI, e.g. tcp://121.41.85.215:19530")
    parser.add_argument("--user", default=DEFAULT_MILVUS_USER, help="Milvus user, e.g. root")
    parser.add_argument("--password", default=DEFAULT_MILVUS_PASSWORD, help="Milvus password")
    parser.add_argument("--token", default=None, help="Milvus token, e.g. root:Milvus")
    parser.add_argument("--db-name", default=DEFAULT_MILVUS_DB_NAME, help="Milvus database name")
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER, choices=["hash", "qwen"])
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=DEFAULT_RECREATE,
        help="Drop and recreate hybrid collections before ingest (default: True)",
    )
    parser.add_argument("--no-recreate", action="store_true", help="Keep existing hybrid collections")
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--qwen-url",
        default=DEFAULT_QWEN_URL,
    )
    return parser.parse_args()


def _load_qwen_key_from_env_files() -> str:
    candidates = [
        BACKEND_ROOT / ".env",
        BACKEND_ROOT.parent / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == "QWEN_API_KEY":
                    key = v.strip().strip('"').strip("'")
                    if key:
                        return key
        except Exception:
            continue
    return ""


def apply_env(args: argparse.Namespace) -> None:
    if args.no_recreate:
        args.recreate = False

    os.environ["MILVUS_ENABLED"] = "true"
    os.environ["MILVUS_URI"] = args.uri
    os.environ["MILVUS_DB_NAME"] = args.db_name
    os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
    os.environ["EMBEDDING_DIM"] = str(args.embedding_dim)

    if args.token:
        os.environ["MILVUS_TOKEN"] = args.token
    elif args.user and args.password:
        os.environ["MILVUS_USER"] = args.user
        os.environ["MILVUS_PASSWORD"] = args.password

    if args.qwen_api_key:
        os.environ["QWEN_API_KEY"] = args.qwen_api_key
    elif not os.environ.get("QWEN_API_KEY"):
        file_key = _load_qwen_key_from_env_files()
        if file_key:
            os.environ["QWEN_API_KEY"] = file_key
    if args.qwen_model:
        os.environ["QWEN_EMBEDDING_MODEL"] = args.qwen_model
    if args.qwen_url:
        os.environ["QWEN_EMBEDDING_URL"] = args.qwen_url

    if args.embedding_provider.lower() == "qwen":
        # Fail fast to avoid silent fallback to hash embeddings.
        qwen_key = os.environ.get("QWEN_API_KEY", "").strip()
        if not qwen_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=qwen but QWEN_API_KEY is empty. "
                "Set env QWEN_API_KEY or pass --qwen-api-key."
            )


def main() -> None:
    args = parse_args()
    apply_env(args)
    print(f"[ingest_to_milvus] uri={args.uri}, user={args.user}, db={args.db_name}")
    print(
        f"[ingest_to_milvus] embedding_provider={args.embedding_provider}, "
        f"embedding_dim={args.embedding_dim}, qwen_model={args.qwen_model}, recreate={args.recreate}"
    )

    from sqlalchemy.orm import Session

    from app.config import settings
    from app.database import Base, engine
    from app.services.ingestion import IngestionService

    Base.metadata.create_all(bind=engine)
    service = IngestionService(recreate_hybrid_collections=args.recreate)
    with Session(engine) as db:
        plans, chunks, facts = service.ingest_all(db)
        db.commit()
    print(f"ingested plans={plans}, chunks={chunks}, facts={facts}")
    print(f"parse output dir: {settings.data_output_dir}")
    print("hybrid ingest completed: direct PDF -> embedding -> Milvus collections")


if __name__ == "__main__":
    main()
