from __future__ import annotations

import argparse
import os


# Fixed defaults for one-click run.
# You can still override them via CLI when needed.
DEFAULT_MILVUS_URI = "tcp://121.41.85.215:19530"
DEFAULT_MILVUS_USER = "root"
DEFAULT_MILVUS_PASSWORD = "Milvus"
DEFAULT_MILVUS_DB_NAME = "default"
DEFAULT_EMBEDDING_PROVIDER = "hash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse PDFs and ingest chunks/facts into SQLite + Milvus.")
    parser.add_argument("--uri", default=DEFAULT_MILVUS_URI, help="Milvus URI, e.g. tcp://121.41.85.215:19530")
    parser.add_argument("--user", default=DEFAULT_MILVUS_USER, help="Milvus user, e.g. root")
    parser.add_argument("--password", default=DEFAULT_MILVUS_PASSWORD, help="Milvus password")
    parser.add_argument("--token", default=None, help="Milvus token, e.g. root:Milvus")
    parser.add_argument("--db-name", default=DEFAULT_MILVUS_DB_NAME, help="Milvus database name")
    parser.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER, choices=["hash", "qwen"])
    parser.add_argument("--qwen-api-key", default=None)
    parser.add_argument("--qwen-model", default="text-embedding-v3")
    parser.add_argument(
        "--qwen-url",
        default="https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
    )
    return parser.parse_args()


def apply_env(args: argparse.Namespace) -> None:
    os.environ["MILVUS_ENABLED"] = "true"
    os.environ["MILVUS_URI"] = args.uri
    os.environ["MILVUS_DB_NAME"] = args.db_name
    os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider

    if args.token:
        os.environ["MILVUS_TOKEN"] = args.token
    elif args.user and args.password:
        os.environ["MILVUS_USER"] = args.user
        os.environ["MILVUS_PASSWORD"] = args.password

    if args.qwen_api_key:
        os.environ["QWEN_API_KEY"] = args.qwen_api_key
    if args.qwen_model:
        os.environ["QWEN_EMBEDDING_MODEL"] = args.qwen_model
    if args.qwen_url:
        os.environ["QWEN_EMBEDDING_URL"] = args.qwen_url


def main() -> None:
    args = parse_args()
    apply_env(args)
    print(
        f"[ingest_to_milvus] uri={args.uri}, user={args.user}, db={args.db_name}, embedding={args.embedding_provider}"
    )

    from sqlalchemy.orm import Session

    from app.config import settings
    from app.database import Base, engine
    from app.services.ingestion import IngestionService

    Base.metadata.create_all(bind=engine)
    service = IngestionService()
    with Session(engine) as db:
        plans, chunks, facts = service.ingest_all(db)
        db.commit()
    print(f"ingested plans={plans}, chunks={chunks}, facts={facts}")
    print(f"parse output dir: {settings.data_output_dir}")


if __name__ == "__main__":
    main()
