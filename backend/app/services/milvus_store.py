from __future__ import annotations

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


class MilvusStore:
    """Optional sink for policy_chunks and policy_facts.

    Collection design for cross-policy retrieval and comparison:
    - policy_chunks: retrievable chunk text + source location + vector embedding
    - policy_facts: structured facts + condition + source + vector embedding
    """

    CHUNK_COLLECTION = "policy_chunks"
    FACT_COLLECTION = "policy_facts"

    def __init__(self) -> None:
        self.enabled = bool(settings.milvus_enabled and MilvusClient and FieldSchema)
        self.dim = settings.embedding_dim
        self.client: MilvusClient | None = None

    def _connection_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"uri": settings.milvus_uri, "db_name": settings.milvus_db_name}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
            return kwargs

        if settings.milvus_user and settings.milvus_password:
            kwargs["user"] = settings.milvus_user
            kwargs["password"] = settings.milvus_password
        return kwargs

    def connect(self) -> None:
        if not self.enabled:
            return
        assert connections is not None
        assert MilvusClient is not None
        kwargs = self._connection_kwargs()
        connections.connect(**kwargs)
        self.client = MilvusClient(**kwargs)

    def ensure_collections(self) -> None:
        if not self.enabled:
            return
        assert FieldSchema is not None
        assert CollectionSchema is not None
        assert DataType is not None
        assert utility is not None

        if not utility.has_collection(self.CHUNK_COLLECTION):
            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="plan_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="plan_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="product_version", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="page_start", dtype=DataType.INT64),
                FieldSchema(name="page_end", dtype=DataType.INT64),
                FieldSchema(name="source_ref", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            coll = Collection(name=self.CHUNK_COLLECTION, schema=schema)
            coll.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            )
            coll.load()

        if not utility.has_collection(self.FACT_COLLECTION):
            fields = [
                FieldSchema(name="fact_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
                FieldSchema(name="plan_id", dtype=DataType.VARCHAR, max_length=64),
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
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            ]
            schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
            coll = Collection(name=self.FACT_COLLECTION, schema=schema)
            coll.create_index(
                field_name="embedding",
                index_params={"index_type": "AUTOINDEX", "metric_type": "COSINE"},
            )
            coll.load()

    def upsert_chunks(self, records: list[dict[str, Any]]) -> None:
        if not self.enabled or not records or self.client is None:
            return
        self.client.upsert(collection_name=self.CHUNK_COLLECTION, data=records)

    def upsert_facts(self, records: list[dict[str, Any]]) -> None:
        if not self.enabled or not records or self.client is None:
            return
        self.client.upsert(collection_name=self.FACT_COLLECTION, data=records)

    @staticmethod
    def now_ts() -> int:
        return int(time.time())
