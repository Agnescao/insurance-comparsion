from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.milvus_hybrid_store import HybridMilvusStore
from app.services.sparse_bm25 import BM25SparseEncoder, tokenize

DEFAULT_SQLITE = "backend/data/app.sqlite3"
DEFAULT_MILVUS_URI = "tcp://121.41.85.215:19530"
DEFAULT_MILVUS_USER = "root"
DEFAULT_MILVUS_PASSWORD = "Milvus"
DEFAULT_MILVUS_DB_NAME = "default"
DEFAULT_CHUNK_COLLECTION = "policy_chunks_hybrid"
DEFAULT_FACT_COLLECTION = "policy_facts_hybrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync parsed SQLite chunks/facts into Milvus hybrid collections.")
    parser.add_argument("--sqlite", default=DEFAULT_SQLITE, help="Path to SQLite file")
    parser.add_argument("--uri", default=DEFAULT_MILVUS_URI, help="Milvus URI, e.g. tcp://121.41.85.215:19530")
    parser.add_argument("--user", default=DEFAULT_MILVUS_USER, help="Milvus user")
    parser.add_argument("--password", default=DEFAULT_MILVUS_PASSWORD, help="Milvus password")
    parser.add_argument("--token", default=None, help="Milvus token, e.g. root:Milvus")
    parser.add_argument("--db-name", default=DEFAULT_MILVUS_DB_NAME, help="Milvus db name")
    parser.add_argument("--chunk-collection", default=DEFAULT_CHUNK_COLLECTION)
    parser.add_argument("--fact-collection", default=DEFAULT_FACT_COLLECTION)
    parser.add_argument("--dim", type=int, default=0, help="Dense embedding dimension, 0 means auto-detect from SQLite")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collections")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
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
        for fmt in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ):
            try:
                return int(datetime.strptime(s, fmt).timestamp())
            except ValueError:
                continue
    return int(time.time())


def _infer_dim(chunk_rows: list[sqlite3.Row], fallback_dim: int = 256) -> int:
    for r in chunk_rows:
        emb = _parse_json(r["embedding"])
        if isinstance(emb, list) and emb:
            try:
                return int(len(emb))
            except Exception:
                continue
    return fallback_dim


def main() -> None:
    args = parse_args()
    print(
        f"[sync_sqlite_to_milvus] sqlite={args.sqlite}, uri={args.uri}, db={args.db_name}, "
        f"chunk_collection={args.chunk_collection}, fact_collection={args.fact_collection}"
    )
    sqlite_path = Path(args.sqlite)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

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

    dim = args.dim if args.dim and args.dim > 0 else _infer_dim(chunk_rows, fallback_dim=256)
    print(f"[sync_sqlite_to_milvus] resolved dense dim={dim}")

    store = HybridMilvusStore(
        uri=args.uri,
        user=args.user,
        password=args.password,
        token=args.token,
        db_name=args.db_name,
        dim=dim,
        chunk_collection=args.chunk_collection,
        fact_collection=args.fact_collection,
        force_enabled=True,
    )
    store.connect()
    store.ensure_collections(recreate=args.recreate)

    # Build BM25 statistics on chunk corpus.
    chunk_tokens = [tokenize(r["text"] or "") for r in chunk_rows]
    bm25 = BM25SparseEncoder(chunk_tokens, k1=args.k1, b=args.b)

    chunk_records: list[dict[str, Any]] = []
    chunk_dense_map: dict[str, list[float]] = {}
    chunk_sparse_map: dict[str, dict[int, float]] = {}

    for r, toks in zip(chunk_rows, chunk_tokens):
        plan = plan_map.get(r["plan_id"], {})
        dense = _parse_json(r["embedding"]) or [0.0] * dim
        if len(dense) > dim:
            dense = dense[:dim]
        elif len(dense) < dim:
            dense = dense + [0.0] * (dim - len(dense))
        sparse = bm25.encode_doc(toks)

        chunk_dense_map[r["chunk_id"]] = dense
        chunk_sparse_map[r["chunk_id"]] = sparse

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
                "dense_embedding": dense,
                "sparse_embedding": sparse,
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
        dense = chunk_dense_map.get(r["source_chunk_id"], [0.0] * dim)
        sparse = chunk_sparse_map.get(r["source_chunk_id"], {})
        if len(dense) > dim:
            dense = dense[:dim]
        elif len(dense) < dim:
            dense = dense + [0.0] * (dim - len(dense))
        fact_records.append(
            {
                "fact_id": r["fact_id"],
                "plan_id": r["plan_id"],
                "plan_name": (plan.get("name") or "")[:255],
                "dimension_key": (r["dimension_key"] or "")[:128],
                "dimension_label": (r["dimension_label"] or "")[:255],
                "value_text": (r["value_text"] or "")[:2048],
                "normalized_value": (r["normalized_value"] or "")[:2048],
                "unit": (r["unit"] or "")[:64],
                "condition_text": (r["condition_text"] or "")[:1024],
                "applicability": (r["applicability"] or "")[:255],
                "source_chunk_id": (r["source_chunk_id"] or "")[:64],
                "source_page": int(r["source_page"] or -1),
                "source_section": (r["source_section"] or "")[:512],
                "confidence": float(r["confidence"] or 0.0),
                "created_at": _to_unix_ts(r["created_at"]),
                "dense_embedding": dense,
                "sparse_embedding": sparse,
            }
        )

    conn.close()

    n_chunks = store.upsert_chunks(chunk_records, batch_size=args.batch_size)
    n_facts = store.upsert_facts(fact_records, batch_size=args.batch_size)

    chunk_stats = store.collection_stats(args.chunk_collection)
    fact_stats = store.collection_stats(args.fact_collection)

    print(f"synced chunks={n_chunks} -> {args.chunk_collection}")
    print(f"synced facts={n_facts} -> {args.fact_collection}")
    print("chunk_stats:", chunk_stats)
    print("fact_stats:", fact_stats)
    print("collections:", store.list_collections())


if __name__ == "__main__":
    main()
