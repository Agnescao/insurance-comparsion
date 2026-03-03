from __future__ import annotations

import logging
import time
from typing import Any

from app.config import settings

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
        connections,
        utility,
    )
except Exception:  # pragma: no cover - optional dependency
    Collection = None
    CollectionSchema = None
    DataType = None
    FieldSchema = None
    MilvusClient = None
    connections = None
    utility = None


class HybridMilvusStore:
    """Milvus store for hybrid retrieval (BM25 sparse + HNSW dense).

    Collections:
    - policy_chunks_hybrid: chunk-level retrievable text and source information.
    - policy_facts_hybrid: structured fact entries for compare/explanation/source tracing.
    """

    def __init__(
        self,
        *,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        token: str | None = None,
        db_name: str | None = None,
        dim: int | None = None,
        chunk_collection: str | None = None,
        fact_collection: str | None = None,
        force_enabled: bool = False,
    ) -> None:
        self.logger = logging.getLogger("uvicorn.error")
        self.uri = uri or settings.milvus_uri
        self.user = user if user is not None else settings.milvus_user
        self.password = password if password is not None else settings.milvus_password
        self.token = token if token is not None else settings.milvus_token
        self.db_name = db_name or settings.milvus_db_name

        self.dim = int(dim or settings.embedding_dim)
        self.chunk_collection = chunk_collection or settings.milvus_chunk_hybrid_collection
        self.fact_collection = fact_collection or settings.milvus_fact_hybrid_collection

        self.enabled = bool((force_enabled or settings.milvus_enabled) and MilvusClient and FieldSchema)
        self.client: MilvusClient | None = None
        self._chunk_varchar_limits: dict[str, int] = {
            "chunk_id": 64,
            "plan_id": 64,
            "plan_name": 255,
            "section_path": 512,
            "source_ref": 512,
            "language": 16,
            "text": 8192,
        }
        self._fact_varchar_limits: dict[str, int] = {
            "fact_id": 64,
            "plan_id": 64,
            "plan_name": 255,
            "dimension_key": 128,
            "dimension_label": 255,
            "value_text": 2048,
            "normalized_value": 2048,
            "unit": 64,
            "condition_text": 1024,
            "applicability": 255,
            "source_chunk_id": 64,
            "source_section": 512,
        }

    def connection_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"uri": self.uri, "db_name": self.db_name}
        if self.token:
            kwargs["token"] = self.token
            return kwargs
        if self.user and self.password:
            kwargs["user"] = self.user
            kwargs["password"] = self.password
        return kwargs

    def connect(self) -> None:
        if not self.enabled:
            return
        assert connections is not None
        assert MilvusClient is not None
        kwargs = self.connection_kwargs()
        connections.connect(alias="default", **kwargs)
        self.client = MilvusClient(**kwargs)

    def list_collections(self) -> list[str]:
        if not self.client:
            return []
        return self.client.list_collections()

    def collection_stats(self, name: str) -> dict[str, Any]:
        if not self.client:
            return {}
        return self.client.get_collection_stats(collection_name=name)

    def ensure_collections(self, recreate: bool = False) -> None:
        if not self.enabled:
            return
        assert utility is not None

        if recreate:
            if utility.has_collection(self.chunk_collection):
                utility.drop_collection(self.chunk_collection)
            if utility.has_collection(self.fact_collection):
                utility.drop_collection(self.fact_collection)

        if not utility.has_collection(self.chunk_collection):
            self.create_chunk_collection()
        if not utility.has_collection(self.fact_collection):
            self.create_fact_collection()

    def create_chunk_collection(self) -> None:
        assert FieldSchema is not None
        assert CollectionSchema is not None
        assert DataType is not None
        assert Collection is not None

        fields = [
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
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        coll = Collection(name=self.chunk_collection, schema=schema)
        coll.create_index(
            field_name="dense_embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        coll.create_index(
            field_name="sparse_embedding",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.1},
            },
        )
        coll.load()

    def create_fact_collection(self) -> None:
        assert FieldSchema is not None
        assert CollectionSchema is not None
        assert DataType is not None
        assert Collection is not None

        fields = [
            FieldSchema(name="fact_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="plan_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="plan_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="dimension_key", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="dimension_label", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="value_text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="normalized_value", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="unit", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="condition_text", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="applicability", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="source_chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="source_page", dtype=DataType.INT64),
            FieldSchema(name="source_section", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
        coll = Collection(name=self.fact_collection, schema=schema)
        coll.create_index(
            field_name="dense_embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        coll.create_index(
            field_name="sparse_embedding",
            index_params={
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.1},
            },
        )
        coll.load()

    def upsert_chunks(self, records: list[dict[str, Any]], batch_size: int = 200) -> int:
        if not self.client or not records:
            return 0
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch, truncated = self._sanitize_varchar_batch(batch, self._chunk_varchar_limits)
            if truncated > 0:
                self.logger.warning(
                    "milvus.sanitize.truncate collection=%s batch_start=%d truncated_fields=%d",
                    self.chunk_collection,
                    i,
                    truncated,
                )
            try:
                self.client.upsert(collection_name=self.chunk_collection, data=batch)
                total += len(batch)
            except Exception as exc:
                self.logger.warning(
                    "milvus.upsert.batch_failed collection=%s batch_start=%d size=%d err=%s",
                    self.chunk_collection,
                    i,
                    len(batch),
                    exc,
                )
                total += self._upsert_rows_resilient(self.chunk_collection, batch, i)
        return total

    def upsert_facts(self, records: list[dict[str, Any]], batch_size: int = 200) -> int:
        if not self.client or not records:
            return 0
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch, truncated = self._sanitize_varchar_batch(batch, self._fact_varchar_limits)
            if truncated > 0:
                self.logger.warning(
                    "milvus.sanitize.truncate collection=%s batch_start=%d truncated_fields=%d",
                    self.fact_collection,
                    i,
                    truncated,
                )
            try:
                self.client.upsert(collection_name=self.fact_collection, data=batch)
                total += len(batch)
            except Exception as exc:
                self.logger.warning(
                    "milvus.upsert.batch_failed collection=%s batch_start=%d size=%d err=%s",
                    self.fact_collection,
                    i,
                    len(batch),
                    exc,
                )
                total += self._upsert_rows_resilient(self.fact_collection, batch, i)
        return total

    @staticmethod
    def now_ts() -> int:
        return int(time.time())

    def _sanitize_varchar_batch(
        self,
        batch: list[dict[str, Any]],
        limits: dict[str, int],
    ) -> tuple[list[dict[str, Any]], int]:
        sanitized: list[dict[str, Any]] = []
        truncated_fields = 0
        for row in batch:
            item = dict(row)
            for key, max_bytes in limits.items():
                if key not in item:
                    continue
                original = item.get(key)
                value = "" if original is None else str(original)
                clipped = self._truncate_utf8(value, max_bytes)
                if clipped != value:
                    truncated_fields += 1
                item[key] = clipped
            sanitized.append(item)
        return sanitized, truncated_fields

    @staticmethod
    def _truncate_utf8(value: str, max_bytes: int) -> str:
        raw = (value or "").encode("utf-8")
        if len(raw) <= max_bytes:
            return value
        clipped = raw[:max_bytes]
        while clipped:
            try:
                return clipped.decode("utf-8")
            except UnicodeDecodeError as exc:
                clipped = clipped[: exc.start]
        return ""

    def _upsert_rows_resilient(self, collection: str, rows: list[dict[str, Any]], batch_start: int) -> int:
        if not self.client:
            return 0
        ok = 0
        for offset, row in enumerate(rows):
            idx = batch_start + offset
            try:
                self.client.upsert(collection_name=collection, data=[row])
                ok += 1
            except Exception as exc:
                row_id = row.get("fact_id") or row.get("chunk_id") or "<unknown>"
                self.logger.error(
                    "milvus.upsert.row_failed collection=%s row_index=%d row_id=%s err=%s",
                    collection,
                    idx,
                    row_id,
                    exc,
                )
        return ok
