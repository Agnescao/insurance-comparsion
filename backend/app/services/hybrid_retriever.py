from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from threading import Lock
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import PolicyChunk
from app.services.embeddings import build_embedding_provider, cosine_similarity
from app.services.sparse_bm25 import term_index, tokenize

try:
    from pymilvus import MilvusClient
except Exception:  # pragma: no cover - optional dependency
    MilvusClient = None


class HybridRetriever:
    def __init__(self) -> None:
        self.logger = logging.getLogger("uvicorn.error")
        self.embedding_provider = build_embedding_provider()
        self.dense_w = settings.retrieval_dense_weight
        self.sparse_w = settings.retrieval_sparse_weight

        self.chunk_collection = settings.milvus_chunk_hybrid_collection
        self.milvus_enabled = bool(settings.milvus_enabled and MilvusClient is not None)
        self._milvus_client: MilvusClient | None = None
        self._milvus_unavailable = False
        self._milvus_lock = Lock()

    def discover_plan_ids(
        self,
        db: Session,
        query: str,
        top_k: int = 3,
        candidate_plan_ids: list[str] | None = None,
    ) -> list[str]:
        if not query.strip():
            return []

        # Preferred path: true hybrid retrieval in Milvus
        milvus_ids = self._discover_plan_ids_milvus(
            query=query,
            top_k=top_k,
            candidate_plan_ids=candidate_plan_ids,
        )
        if milvus_ids:
            return milvus_ids

        # Fallback path: local hybrid scoring from SQLite chunks
        return self._discover_plan_ids_sqlite(
            db=db,
            query=query,
            top_k=top_k,
            candidate_plan_ids=candidate_plan_ids,
        )

    def retrieve_plan_evidence(
        self,
        db: Session,
        query: str,
        plan_ids: list[str],
        per_plan_k: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        plan_ids = [pid for pid in plan_ids if isinstance(pid, str) and pid.strip()]
        if not query.strip() or not plan_ids:
            return {pid: [] for pid in plan_ids}

        milvus = self._retrieve_plan_evidence_milvus(query=query, plan_ids=plan_ids, per_plan_k=per_plan_k)
        if any(milvus.values()):
            return milvus
        return self._retrieve_plan_evidence_sqlite(db=db, query=query, plan_ids=plan_ids, per_plan_k=per_plan_k)

    def _discover_plan_ids_milvus(
        self,
        query: str,
        top_k: int,
        candidate_plan_ids: list[str] | None = None,
    ) -> list[str]:
        if not self.milvus_enabled:
            return []

        client = self._get_milvus_client()
        if client is None:
            return []

        try:
            collections = set(client.list_collections())
            if self.chunk_collection not in collections:
                return []

            dense_query = self.embedding_provider.embed([query])[0]
            sparse_query = self._sparse_query_vector(query)
            if not dense_query and not sparse_query:
                return []

            limit = max(20, top_k * 10)
            expr = self._build_filter_expr(candidate_plan_ids)

            common_kwargs: dict[str, Any] = {
                "collection_name": self.chunk_collection,
                "limit": limit,
                "output_fields": ["chunk_id", "plan_id"],
            }
            if expr:
                common_kwargs["filter"] = expr

            dense_hits_raw = client.search(
                **common_kwargs,
                data=[dense_query],
                anns_field="dense_embedding",
                search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            )
            sparse_hits_raw = client.search(
                **common_kwargs,
                data=[sparse_query],
                anns_field="sparse_embedding",
                search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
            )

            dense_scores = self._hits_to_chunk_scores(dense_hits_raw)
            sparse_scores = self._hits_to_chunk_scores(sparse_hits_raw)
            fused = self._fuse_scores(dense_scores, sparse_scores)
            if not fused:
                return []

            plan_scores: dict[str, list[float]] = defaultdict(list)
            for _, item in fused.items():
                plan_scores[item["plan_id"]].append(item["score"])

            scored: list[tuple[str, float]] = []
            for plan_id, vals in plan_scores.items():
                vals.sort(reverse=True)
                scored.append((plan_id, sum(vals[:3])))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [pid for pid, _ in scored[: max(2, top_k)]]
        except Exception:
            # Degrade gracefully to local fallback.
            return []

    def _retrieve_plan_evidence_milvus(
        self,
        query: str,
        plan_ids: list[str],
        per_plan_k: int,
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {pid: [] for pid in plan_ids}
        if not self.milvus_enabled:
            return out

        client = self._get_milvus_client()
        if client is None:
            return out

        try:
            collections = set(client.list_collections())
            if self.chunk_collection not in collections:
                return out

            dense_query = self.embedding_provider.embed([query])[0]
            sparse_query = self._sparse_query_vector(query)
            if not dense_query and not sparse_query:
                return out

            limit = max(40, per_plan_k * max(2, len(plan_ids)) * 8)
            expr = self._build_filter_expr(plan_ids)
            common_kwargs: dict[str, Any] = {
                "collection_name": self.chunk_collection,
                "limit": limit,
                "output_fields": ["chunk_id", "plan_id", "section_path", "page_start", "text", "source_ref"],
            }
            if expr:
                common_kwargs["filter"] = expr

            dense_hits_raw = client.search(
                **common_kwargs,
                data=[dense_query],
                anns_field="dense_embedding",
                search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            )
            sparse_hits_raw = client.search(
                **common_kwargs,
                data=[sparse_query],
                anns_field="sparse_embedding",
                search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
            )

            dense_scores = self._hits_to_chunk_scores(dense_hits_raw)
            sparse_scores = self._hits_to_chunk_scores(sparse_hits_raw)
            fused = self._fuse_scores(dense_scores, sparse_scores)
            ranked = sorted(fused.items(), key=lambda x: float(x[1]["score"]), reverse=True)

            for _, row in ranked:
                pid = str(row.get("plan_id") or "")
                if pid not in out or len(out[pid]) >= per_plan_k:
                    continue
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                out[pid].append(
                    {
                        "score": float(row.get("score") or 0.0),
                        "quote": text[:300],
                        "page": self._safe_int(row.get("page_start")),
                        "section": row.get("section_path") or None,
                        "source_ref": row.get("source_ref") or None,
                        "from": "milvus",
                    }
                )
                if all(len(v) >= per_plan_k for v in out.values()):
                    break
            return out
        except Exception:
            return out

    def _discover_plan_ids_sqlite(
        self,
        db: Session,
        query: str,
        top_k: int = 3,
        candidate_plan_ids: list[str] | None = None,
    ) -> list[str]:
        stmt = select(PolicyChunk)
        if candidate_plan_ids:
            stmt = stmt.where(PolicyChunk.plan_id.in_(candidate_plan_ids))
        chunks = db.execute(stmt).scalars().all()
        if not chunks:
            return []

        docs: list[dict[str, Any]] = []
        for c in chunks:
            tokens = tokenize(c.text)
            if not tokens:
                continue
            docs.append(
                {
                    "chunk_id": c.chunk_id,
                    "plan_id": c.plan_id,
                    "tokens": tokens,
                    "embedding": c.embedding or [],
                }
            )
        if not docs:
            return []

        q_tokens = tokenize(query)
        q_vec = self.embedding_provider.embed([query])[0]

        bm25_scores = self._bm25_scores(docs, q_tokens)
        dense_scores = self._dense_scores(docs, q_vec)

        bm25_norm = self._min_max_norm(bm25_scores)
        dense_norm = self._min_max_norm(dense_scores)

        by_plan: dict[str, list[float]] = defaultdict(list)
        for d in docs:
            cid = d["chunk_id"]
            score = self.dense_w * dense_norm.get(cid, 0.0) + self.sparse_w * bm25_norm.get(cid, 0.0)
            by_plan[d["plan_id"]].append(score)

        plan_scores: list[tuple[str, float]] = []
        for pid, values in by_plan.items():
            values.sort(reverse=True)
            plan_scores.append((pid, sum(values[:3])))

        plan_scores.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in plan_scores[: max(2, top_k)]]

    def _retrieve_plan_evidence_sqlite(
        self,
        db: Session,
        query: str,
        plan_ids: list[str],
        per_plan_k: int,
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {pid: [] for pid in plan_ids}
        stmt = (
            select(PolicyChunk)
            .where(PolicyChunk.plan_id.in_(plan_ids))
            .order_by(PolicyChunk.page_start.asc(), PolicyChunk.paragraph_index.asc())
        )
        chunks = db.execute(stmt).scalars().all()
        if not chunks:
            return out

        docs: list[dict[str, Any]] = []
        for c in chunks:
            tokens = tokenize(c.text)
            if not tokens:
                continue
            docs.append(
                {
                    "chunk_id": c.chunk_id,
                    "plan_id": c.plan_id,
                    "tokens": tokens,
                    "embedding": c.embedding or [],
                    "text": c.text or "",
                    "page_start": c.page_start,
                    "section_path": c.section_path,
                }
            )
        if not docs:
            return out

        q_tokens = tokenize(query)
        q_vec = self.embedding_provider.embed([query])[0]

        bm25_scores = self._bm25_scores(docs, q_tokens)
        dense_scores = self._dense_scores(docs, q_vec)
        bm25_norm = self._min_max_norm(bm25_scores)
        dense_norm = self._min_max_norm(dense_scores)

        scored_rows: list[dict[str, Any]] = []
        for d in docs:
            cid = d["chunk_id"]
            score = self.dense_w * dense_norm.get(cid, 0.0) + self.sparse_w * bm25_norm.get(cid, 0.0)
            scored_rows.append(
                {
                    "plan_id": d["plan_id"],
                    "score": float(score),
                    "quote": str(d["text"]).strip()[:300],
                    "page": d.get("page_start"),
                    "section": d.get("section_path"),
                    "from": "sqlite",
                }
            )
        scored_rows.sort(key=lambda x: x["score"], reverse=True)

        for row in scored_rows:
            pid = str(row["plan_id"])
            if pid not in out or len(out[pid]) >= per_plan_k:
                continue
            if not row["quote"]:
                continue
            out[pid].append(row)
            if all(len(v) >= per_plan_k for v in out.values()):
                break
        return out

    def _get_milvus_client(self) -> MilvusClient | None:
        if self._milvus_client is not None:
            return self._milvus_client
        if not self.milvus_enabled or MilvusClient is None:
            return None
        if self._milvus_unavailable:
            return None

        with self._milvus_lock:
            if self._milvus_client is not None:
                return self._milvus_client
            if self._milvus_unavailable:
                return None

            kwargs: dict[str, Any] = {
                "uri": settings.milvus_uri,
                "db_name": settings.milvus_db_name,
            }
            if settings.milvus_token:
                kwargs["token"] = settings.milvus_token
            elif settings.milvus_user and settings.milvus_password:
                kwargs["user"] = settings.milvus_user
                kwargs["password"] = settings.milvus_password
            try:
                self._milvus_client = MilvusClient(**kwargs)
                self.logger.info("retriever.milvus.connect.ok uri=%s db=%s", settings.milvus_uri, settings.milvus_db_name)
                return self._milvus_client
            except Exception as exc:
                self._milvus_unavailable = True
                self.logger.warning("retriever.milvus.connect.failed uri=%s err=%s", settings.milvus_uri, exc)
                return None

    def _build_filter_expr(self, candidate_plan_ids: list[str] | None) -> str | None:
        if not candidate_plan_ids:
            return None
        escaped = [pid.replace("\\", "\\\\").replace('"', '\\"') for pid in candidate_plan_ids if pid]
        if not escaped:
            return None
        joined = '","'.join(escaped)
        return f'plan_id in ["{joined}"]'

    def _sparse_query_vector(self, query: str) -> dict[int, float]:
        tokens = tokenize(query)
        if not tokens:
            return {}
        tf = Counter(tokens)
        return {term_index(tok): float(cnt) for tok, cnt in tf.items() if cnt > 0}

    def _hits_to_chunk_scores(self, hits_raw: Any) -> dict[str, dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(hits_raw, list) and hits_raw:
            if isinstance(hits_raw[0], list):
                rows = [h for h in hits_raw[0] if isinstance(h, dict)]
            elif isinstance(hits_raw[0], dict):
                rows = [h for h in hits_raw if isinstance(h, dict)]

        scores: dict[str, dict[str, Any]] = {}
        for row in rows:
            entity = row.get("entity", {}) if isinstance(row.get("entity"), dict) else {}
            chunk_id = row.get("id") or entity.get("chunk_id") or row.get("chunk_id")
            plan_id = entity.get("plan_id") or row.get("plan_id")
            raw_score = row.get("distance")
            if raw_score is None:
                raw_score = row.get("score")
            if chunk_id is None or plan_id is None or raw_score is None:
                continue

            cid = str(chunk_id)
            score = float(raw_score)
            prev = scores.get(cid)
            if prev is None or score > float(prev["score"]):
                scores[cid] = {
                    "plan_id": str(plan_id),
                    "score": score,
                    "text": entity.get("text") or row.get("text"),
                    "page_start": entity.get("page_start") or row.get("page_start"),
                    "section_path": entity.get("section_path") or row.get("section_path"),
                    "source_ref": entity.get("source_ref") or row.get("source_ref"),
                }
        return scores

    def _fuse_scores(
        self,
        dense_scores: dict[str, dict[str, Any]],
        sparse_scores: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not dense_scores and not sparse_scores:
            return {}

        dense_norm = self._min_max_norm({k: v["score"] for k, v in dense_scores.items()})
        sparse_norm = self._min_max_norm({k: v["score"] for k, v in sparse_scores.items()})

        chunk_ids = set(dense_scores) | set(sparse_scores)
        fused: dict[str, dict[str, Any]] = {}
        for cid in chunk_ids:
            plan_id = None
            if cid in dense_scores:
                plan_id = dense_scores[cid]["plan_id"]
            elif cid in sparse_scores:
                plan_id = sparse_scores[cid]["plan_id"]
            if not plan_id:
                continue

            meta = dense_scores.get(cid) or sparse_scores.get(cid) or {}
            fused[cid] = {
                "plan_id": plan_id,
                "score": self.dense_w * dense_norm.get(cid, 0.0) + self.sparse_w * sparse_norm.get(cid, 0.0),
                "text": meta.get("text"),
                "page_start": meta.get("page_start"),
                "section_path": meta.get("section_path"),
                "source_ref": meta.get("source_ref"),
            }
        return fused

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _bm25_scores(self, docs: list[dict[str, Any]], q_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        if not q_tokens:
            return {d["chunk_id"]: 0.0 for d in docs}
        N = len(docs)
        avgdl = sum(len(d["tokens"]) for d in docs) / max(1, N)

        df: dict[str, int] = defaultdict(int)
        for d in docs:
            for t in set(d["tokens"]):
                df[t] += 1

        idf: dict[str, float] = {}
        for t in q_tokens:
            n_t = df.get(t, 0)
            idf[t] = math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))

        scores: dict[str, float] = {}
        for d in docs:
            tf = Counter(d["tokens"])
            dl = len(d["tokens"])
            score = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + k1 * (1 - b + b * dl / max(1e-6, avgdl))
                score += idf.get(t, 0.0) * (f * (k1 + 1)) / max(1e-6, denom)
            scores[d["chunk_id"]] = score
        return scores

    def _dense_scores(self, docs: list[dict[str, Any]], q_vec: list[float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for d in docs:
            emb = d["embedding"] or []
            scores[d["chunk_id"]] = float(cosine_similarity(q_vec, emb)) if q_vec and emb else 0.0
        return scores

    def _min_max_norm(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
