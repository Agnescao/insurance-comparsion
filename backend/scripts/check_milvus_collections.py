from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_MILVUS_URI = "tcp://121.41.85.215:19530"
DEFAULT_MILVUS_USER = "root"
DEFAULT_MILVUS_PASSWORD = "Milvus"
DEFAULT_MILVUS_DB_NAME = "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show Milvus collections and row counts.")
    parser.add_argument("--uri", default=DEFAULT_MILVUS_URI)
    parser.add_argument("--user", default=DEFAULT_MILVUS_USER)
    parser.add_argument("--password", default=DEFAULT_MILVUS_PASSWORD)
    parser.add_argument("--token", default=None)
    parser.add_argument("--db-name", default=DEFAULT_MILVUS_DB_NAME)
    return parser.parse_args()


def _client(args: argparse.Namespace) -> MilvusClient:
    kwargs: dict[str, Any] = {"uri": args.uri, "db_name": args.db_name}
    if args.token:
        kwargs["token"] = args.token
    else:
        kwargs["user"] = args.user
        kwargs["password"] = args.password
    return MilvusClient(**kwargs)


def main() -> None:
    args = parse_args()
    client = _client(args)
    collections = client.list_collections()
    print("collections:", collections)
    for name in collections:
        stats = client.get_collection_stats(collection_name=name)
        print(f"{name} -> {stats}")
        try:
            schema = client.describe_collection(collection_name=name)
            print(f"{name} schema -> {schema}")
        except Exception as exc:
            print(f"{name} schema -> <failed: {type(exc).__name__}: {exc}>")


if __name__ == "__main__":
    main()
