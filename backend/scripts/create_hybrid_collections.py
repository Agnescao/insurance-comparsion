from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.services.milvus_hybrid_store import HybridMilvusStore

DEFAULT_MILVUS_URI = "tcp://121.41.85.215:19530"
DEFAULT_MILVUS_USER = "root"
DEFAULT_MILVUS_PASSWORD = "Milvus"
DEFAULT_MILVUS_DB_NAME = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Milvus hybrid collections for chunks and facts.")
    parser.add_argument("--uri", default=DEFAULT_MILVUS_URI)
    parser.add_argument("--user", default=DEFAULT_MILVUS_USER)
    parser.add_argument("--password", default=DEFAULT_MILVUS_PASSWORD)
    parser.add_argument("--token", default=settings.milvus_token)
    parser.add_argument("--db-name", default=DEFAULT_MILVUS_DB_NAME)
    parser.add_argument("--dim", type=int, default=settings.embedding_dim)
    parser.add_argument("--chunk-collection", default=settings.milvus_chunk_hybrid_collection)
    parser.add_argument("--fact-collection", default=settings.milvus_fact_hybrid_collection)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"[create_hybrid_collections] uri={args.uri}, user={args.user}, db={args.db_name}, "
        f"chunk={args.chunk_collection}, fact={args.fact_collection}"
    )
    store = HybridMilvusStore(
        uri=args.uri,
        user=args.user,
        password=args.password,
        token=args.token,
        db_name=args.db_name,
        dim=args.dim,
        chunk_collection=args.chunk_collection,
        fact_collection=args.fact_collection,
        force_enabled=True,
    )
    store.connect()
    store.ensure_collections(recreate=args.recreate)

    print("collections:", store.list_collections())
    print(args.chunk_collection, "stats:", store.collection_stats(args.chunk_collection))
    print(args.fact_collection, "stats:", store.collection_stats(args.fact_collection))


if __name__ == "__main__":
    main()
