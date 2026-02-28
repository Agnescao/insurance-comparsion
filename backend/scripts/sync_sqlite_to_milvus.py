from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync parsed SQLite chunks/facts into Milvus.")
    parser.add_argument("--sqlite", default="backend/data/app.sqlite3", help="Path to SQLite file")
    parser.add_argument("--uri", required=True, help="Milvus URI, e.g. tcp://121.41.85.215:19530")
    parser.add_argument("--user", default="root", help="Milvus user")
    parser.add_argument("--password", default="Milvus", help="Milvus password")
    parser.add_argument("--token", default=None, help="Milvus token, e.g. root:Milvus")
    parser.add_argument("--db-name", default="default", help="Milvus db name")
    parser.add_argument("--chunk-collection", default="policy_chunks_hybrid")
    parser.add_argument("--fact-collection", default="policy_facts_hybrid")
    parser.add_argument("--dim", type=int, default=256, help="Dense embedding dimension")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collections")
    return parser.parse_args()


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _to_unix_ts(v: Any) -> int:
    if v is None:
        return int(time.time())
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip().replace("Z", "")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return int(datetime.strptime(s, fmt).timestamp())
            except ValueError:
                continue
    return int(time.time())


def _connect_client(args: argparse.Namespace) -> MilvusClient:
    kwargs: dict[str, Any] = {"uri": args.uri, "db_name": args.db_name}
    if args.token:
        kwargs["token"] = args.token
    else:
        kwargs["user"] = args.user
        kwargs["password"] = args.password

    connections.connect(alias="default", **kwargs)
    return MilvusClient(**kwargs)


def _ensure_collections(args: argparse.Namespace) -> None:
    if args.recreate:
        if utility.has_collection(args.chunk_collection):
            utility.drop_collection(args.chunk_collection)
        if utility.has_collection(args.fact_collection):
            utility.drop_collection(args.fact_collection)

    if not utility.has_collection(args.chunk_collection):
        chunk_fields = [
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="plan_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="plan_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="source_ref", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dim),
        ]
        chunk_schema = CollectionSchema(fields=chunk_fields, enable_dynamic_field=True)
        chunk_coll = Collection(name=args.chunk_collection, schema=chunk_schema)
        chunk_coll.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        )
        chunk_coll.load()

    if not utility.has_collection(args.fact_collection):
        fact_fields = [
            FieldSchema(name="fact_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="plan_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="plan_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="dimension_key", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="dimension_label", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="value_text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="normalized_value", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="unit", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="condition_text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="applicability", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_page", dtype=DataType.INT64),
            FieldSchema(name="source_section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=args.dim),
        ]
        fact_schema = CollectionSchema(fields=fact_fields, enable_dynamic_field=True)
        fact_coll = Collection(name=args.fact_collection, schema=fact_schema)
        fact_coll.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        )
        fact_coll.load()


def _upsert_batches(client: MilvusClient, collection_name: str, records: list[dict[str, Any]], batch_size: int) -> int:
    total = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        client.upsert(collection_name=collection_name, data=batch)
        total += len(batch)
    return total


def main() -> None:
    args = parse_args()
    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    client = _connect_client(args)
    _ensure_collections(args)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    plans = cur.execute("SELECT plan_id, name, source_file, language FROM plans").fetchall()
    plan_map: dict[str, dict[str, Any]] = {r["plan_id"]: dict(r) for r in plans}

    chunk_rows = cur.execute(
        """
        SELECT chunk_id, plan_id, section_path, page_start, page_end, text, embedding, created_at
        FROM policy_chunks
        """
    ).fetchall()
    chunk_records: list[dict[str, Any]] = []
    chunk_embedding_map: dict[str, list[float]] = {}
    for r in chunk_rows:
        plan = plan_map.get(r["plan_id"], {})
        emb = _parse_json(r["embedding"]) or [0.0] * args.dim
        if len(emb) != args.dim:
            emb = [0.0] * args.dim
        chunk_embedding_map[r["chunk_id"]] = emb
        chunk_records.append(
            {
                "chunk_id": r["chunk_id"],
                "plan_id": r["plan_id"],
                "plan_name": (plan.get("name") or "")[:255],
                "section_path": (r["section_path"] or "")[:512],
                "page_start": int(r["page_start"] or -1),
                "page_end": int(r["page_end"] or -1),
                "source_ref": f"{plan.get('source_file', '')}#p{int(r['page_start'] or 0)}"[:512],
                "language": (plan.get("language") or "zh")[:16],
                "text": (r["text"] or "")[:8192],
                "created_at": _to_unix_ts(r["created_at"]),
                "embedding": emb,
            }
        )

    fact_rows = cur.execute(
        """
        SELECT fact_id, plan_id, dimension_key, dimension_label, value_text, normalized_value, unit,
               condition_text, applicability, source_chunk_id, source_page, source_section, confidence, created_at
        FROM policy_facts
        """
    ).fetchall()
    fact_records: list[dict[str, Any]] = []
    for r in fact_rows:
        plan = plan_map.get(r["plan_id"], {})
        emb = chunk_embedding_map.get(r["source_chunk_id"], [0.0] * args.dim)
        if len(emb) != args.dim:
            emb = [0.0] * args.dim
        fact_records.append(
            {
                "fact_id": r["fact_id"],
                "plan_id": r["plan_id"],
                "plan_name": (plan.get("name") or "")[:255],
                "dimension_key": (r["dimension_key"] or "")[:128],
                "dimension_label": (r["dimension_label"] or "")[:255],
                "value_text": (r["value_text"] or "")[:2048],
                "normalized_value": (r["normalized_value"] or "")[:1024],
                "unit": (r["unit"] or "")[:64],
                "condition_text": (r["condition_text"] or "")[:1024],
                "applicability": (r["applicability"] or "")[:255],
                "source_chunk_id": (r["source_chunk_id"] or "")[:64],
                "source_page": int(r["source_page"] or -1),
                "source_section": (r["source_section"] or "")[:512],
                "confidence": float(r["confidence"] or 0.0),
                "created_at": _to_unix_ts(r["created_at"]),
                "embedding": emb,
            }
        )

    conn.close()

    n_chunks = _upsert_batches(client, args.chunk_collection, chunk_records, args.batch_size)
    n_facts = _upsert_batches(client, args.fact_collection, fact_records, args.batch_size)

    print(f"synced chunks={n_chunks} -> {args.chunk_collection}")
    print(f"synced facts={n_facts} -> {args.fact_collection}")
    print("collections:", client.list_collections())


if __name__ == "__main__":
    main()
