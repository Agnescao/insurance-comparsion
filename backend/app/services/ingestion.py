from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Plan, PolicyChunk, PolicyFact
from app.services.chunking import ChunkDoc, HybridChunker
from app.services.embeddings import build_embedding_provider
from app.services.fact_extractor import FactExtractor, FactRecord
from app.services.llm_fact_extractor import LLMFactExtractor
from app.services.milvus_hybrid_store import HybridMilvusStore
from app.services.parser import PDFParser, ParsedPolicy
from app.services.sparse_bm25 import BM25SparseEncoder, tokenize


def _to_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


@dataclass
class PreparedPolicyDoc:
    pdf_path: Path
    parsed: ParsedPolicy
    chunk_docs: list[ChunkDoc]
    facts: list[FactRecord]


class IngestionService:
    def __init__(self, *, recreate_hybrid_collections: bool = False) -> None:
        self.logger = logging.getLogger("uvicorn.error")
        self.fact_extractor_mode = (settings.fact_extractor_mode or "llm").strip().lower()
        self.milvus = HybridMilvusStore()
        self._milvus_ready = False
        self._recreate_hybrid_collections = recreate_hybrid_collections
        if self.milvus.enabled:
            try:
                self.milvus.connect()
            except Exception as exc:
                self.logger.warning("ingest.milvus.disabled reason=%s", exc)
                self.milvus.enabled = False

    def ingest_all(self, db: Session) -> tuple[int, int, int]:
        pdf_paths = sorted(Path(settings.data_dir).glob("*.pdf"))
        plans_processed = 0
        chunks_written = 0
        facts_written = 0
        workers = max(1, int(settings.ingest_parallel_workers))
        self.logger.info(
            "ingest.all.start docs=%d workers=%d fact_extractor_mode=%s",
            len(pdf_paths),
            workers,
            self.fact_extractor_mode,
        )

        if workers == 1 or len(pdf_paths) <= 1:
            for pdf_path in pdf_paths:
                prepared = self._prepare_doc(pdf_path)
                p_count, c_count, f_count = self._persist_prepared(db, prepared)
                plans_processed += p_count
                chunks_written += c_count
                facts_written += f_count
            return plans_processed, chunks_written, facts_written

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._prepare_doc, pdf_path): pdf_path for pdf_path in pdf_paths}
            for future in as_completed(futures):
                prepared = future.result()
                p_count, c_count, f_count = self._persist_prepared(db, prepared)
                plans_processed += p_count
                chunks_written += c_count
                facts_written += f_count

        self.logger.info(
            "ingest.all.done plans=%d chunks=%d facts=%d",
            plans_processed,
            chunks_written,
            facts_written,
        )
        return plans_processed, chunks_written, facts_written

    def ingest_one(self, db: Session, pdf_path: Path) -> tuple[int, int, int]:
        prepared = self._prepare_doc(pdf_path)
        return self._persist_prepared(db, prepared)

    def _build_fact_extractor(self):
        # if self.fact_extractor_mode == "rule":
        #     return FactExtractor()
        if self.fact_extractor_mode in {"llm", "hybrid"}:
            return LLMFactExtractor(mode=self.fact_extractor_mode, fallback=FactExtractor())
        self.logger.info("ingest.fact_extractor.unknown_mode mode=%s fallback=rule", self.fact_extractor_mode)
        return FactExtractor()

    def _prepare_doc(self, pdf_path: Path) -> PreparedPolicyDoc:
        t0 = perf_counter()
        parser = PDFParser()
        embedding_provider = build_embedding_provider()
        chunker = HybridChunker(embedding_provider)
        fact_extractor = self._build_fact_extractor()

        parsed = parser.parse(pdf_path)
        chunk_docs = chunker.chunk_policy(parsed)
        facts = fact_extractor.extract_from_chunks(
            chunk_docs,
            plan_name=parsed.plan_name,
            source_file=str(pdf_path),
        )
        self.logger.info(
            "ingest.prepare.done file=%s plan=%s chunks=%d facts=%d elapsed=%.3fs",
            pdf_path.name,
            parsed.plan_name,
            len(chunk_docs),
            len(facts),
            perf_counter() - t0,
        )
        return PreparedPolicyDoc(pdf_path=pdf_path, parsed=parsed, chunk_docs=chunk_docs, facts=facts)

    def _persist_prepared(self, db: Session, prepared: PreparedPolicyDoc) -> tuple[int, int, int]:
        pdf_path = prepared.pdf_path
        parsed = prepared.parsed
        chunk_docs = prepared.chunk_docs
        facts = prepared.facts

        existing = db.execute(select(Plan).where(Plan.source_file == str(pdf_path))).scalar_one_or_none()
        if existing:
            plan = existing
            plan.name = parsed.plan_name
            db.execute(delete(PolicyFact).where(PolicyFact.plan_id == plan.plan_id))
            db.execute(delete(PolicyChunk).where(PolicyChunk.plan_id == plan.plan_id))
        else:
            plan = Plan(name=parsed.plan_name, source_file=str(pdf_path), language="zh")
            db.add(plan)
            db.flush()

        chunks: list[PolicyChunk] = []
        for chunk_doc in chunk_docs:
            chunk = PolicyChunk(
                chunk_id=uuid4().hex,
                plan_id=plan.plan_id,
                section_path=chunk_doc.section_path,
                page_start=chunk_doc.page_start,
                page_end=chunk_doc.page_end,
                paragraph_index=chunk_doc.paragraph_index,
                text=chunk_doc.text,
                token_count=max(1, len(chunk_doc.text) // 4),
                embedding=chunk_doc.embedding,
                metadata_json={"source_file": str(pdf_path)},
            )
            db.add(chunk)
            chunks.append(chunk)
        db.flush()

        dim_counts = Counter(f.dimension_key for f in facts)
        self.logger.info(
            "ingest.facts.extracted plan=%s fact_count=%d dim_counts=%s",
            parsed.plan_name,
            len(facts),
            dict(dim_counts),
        )
        premium_terms = sorted(
            {
                int(term)
                for fact in facts
                if fact.dimension_key == "premium_payment"
                for term in (fact.metadata_json or {}).get("payment_terms_years", [])
                if isinstance(term, int) or (isinstance(term, str) and str(term).isdigit())
            }
        )
        if premium_terms:
            self.logger.info("ingest.facts.premium_terms plan=%s terms=%s", parsed.plan_name, premium_terms)

        fact_models: list[PolicyFact] = []
        chunk_by_page: dict[int, PolicyChunk] = {c.page_start or -1: c for c in chunks}
        for fact in facts:
            source_chunk = chunk_by_page.get(fact.source_page or -1)
            metadata = dict(fact.metadata_json or {})
            metadata["source_file"] = str(pdf_path)
            model = PolicyFact(
                fact_id=uuid4().hex,
                plan_id=plan.plan_id,
                dimension_key=fact.dimension_key,
                dimension_label=fact.dimension_label,
                value_text=fact.value_text,
                normalized_value=fact.normalized_value,
                numeric_value=fact.numeric_value,
                unit=fact.unit,
                condition_text=fact.condition_text,
                source_chunk_id=source_chunk.chunk_id if source_chunk else None,
                source_page=fact.source_page,
                source_section=fact.source_section,
                source_quote=fact.source_quote,
                confidence=fact.confidence,
                metadata_json=metadata,
            )
            db.add(model)
            fact_models.append(model)

        db.flush()
        self._dump_parse_output(pdf_path=pdf_path, parsed=parsed, chunk_docs=chunk_docs, facts=facts)
        self._sync_milvus(plan, chunks, fact_models)
        return 1, len(chunks), len(fact_models)

    def _dump_parse_output(
        self,
        pdf_path: Path,
        parsed: ParsedPolicy,
        chunk_docs: list[ChunkDoc],
        facts: list[FactRecord],
    ) -> None:
        if not settings.dump_parse_output:
            return

        out_root = Path(settings.data_output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        doc_dir = out_root / pdf_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        # Always rewrite output artifacts from scratch for each ingestion run.
        for file_name in ("document.md", "facts.json", "fats.json", "facts_grouped.json"):
            fpath = doc_dir / file_name
            if fpath.exists():
                fpath.unlink()

        parsed_json = {
            "plan_name": parsed.plan_name,
            "source_file": parsed.source_file,
            "pages": [
                {
                    "page_number": p.page_number,
                    "markdown": p.markdown,
                    "layout": _to_jsonable(p.layout),
                }
                for p in parsed.pages
            ],
        }
        (doc_dir / "parsed.json").write_text(json.dumps(parsed_json, ensure_ascii=False, indent=2), encoding="utf-8")

        markdown_lines: list[str] = []
        for p in parsed.pages:
            markdown_lines.append(f"\n\n## Page {p.page_number}\n")
            markdown_lines.append(p.markdown or "")
        (doc_dir / "document.md").write_text("".join(markdown_lines), encoding="utf-8")

        chunks_json = [
            {
                "text": c.text,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section_path": c.section_path,
                "paragraph_index": c.paragraph_index,
                "embedding_dim": len(c.embedding or []),
            }
            for c in chunk_docs
        ]
        (doc_dir / "chunks.json").write_text(json.dumps(chunks_json, ensure_ascii=False, indent=2), encoding="utf-8")

        facts_json = [
            {
                "dimension_key": f.dimension_key,
                "dimension_label": f.dimension_label,
                "value_text": f.value_text,
                "normalized_value": f.normalized_value,
                "numeric_value": f.numeric_value,
                "unit": f.unit,
                "condition_text": f.condition_text,
                "confidence": f.confidence,
                "source_page": f.source_page,
                "source_section": f.source_section,
                "source_quote": f.source_quote,
                "metadata_json": f.metadata_json,
            }
            for f in facts
        ]
        facts_payload = json.dumps(facts_json, ensure_ascii=False, indent=2)
        (doc_dir / "facts.json").write_text(facts_payload, encoding="utf-8")
        (doc_dir / "fats.json").write_text(facts_payload, encoding="utf-8")
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in facts_json:
            key = str(row.get("dimension_key") or "")
            if not key:
                continue
            grouped.setdefault(key, []).append(row)
        (doc_dir / "facts_grouped.json").write_text(
            json.dumps(grouped, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _sync_milvus(self, plan: Plan, chunks: list[PolicyChunk], facts: list[PolicyFact]) -> None:
        if not self.milvus.enabled:
            return

        detected_dim = settings.embedding_dim
        for c in chunks:
            emb = c.embedding or []
            if emb:
                detected_dim = len(emb)
                break

        if not self._milvus_ready:
            self.milvus.dim = detected_dim
            self.milvus.ensure_collections(recreate=self._recreate_hybrid_collections)
            self._milvus_ready = True
            self._recreate_hybrid_collections = False

        target_dim = int(self.milvus.dim)

        def _norm_dense(v: list[float] | None) -> list[float]:
            data = list(v or [])
            if len(data) > target_dim:
                return data[:target_dim]
            if len(data) < target_dim:
                return data + [0.0] * (target_dim - len(data))
            return data

        chunk_tokens = [tokenize(c.text or "") for c in chunks]
        bm25 = BM25SparseEncoder(chunk_tokens, k1=1.5, b=0.75)

        chunk_embedding_map: dict[str, list[float]] = {}
        chunk_sparse_map: dict[str, dict[int, float]] = {}
        for c, toks in zip(chunks, chunk_tokens):
            chunk_embedding_map[c.chunk_id] = _norm_dense(c.embedding)
            chunk_sparse_map[c.chunk_id] = bm25.encode_doc(toks)

        chunk_records = [
            {
                "chunk_id": c.chunk_id,
                "plan_id": plan.plan_id,
                "plan_name": plan.name,
                "section_path": c.section_path or "",
                "page_start": c.page_start or -1,
                "page_end": c.page_end or -1,
                "source_ref": f"{plan.source_file}#p{c.page_start or 0}",
                "language": plan.language or "zh",
                "text": c.text[:8192],
                "created_at": self.milvus.now_ts(),
                "dense_embedding": chunk_embedding_map[c.chunk_id],
                "sparse_embedding": chunk_sparse_map.get(c.chunk_id, {}),
            }
            for c in chunks
        ]

        fact_records = [
            {
                "fact_id": f.fact_id,
                "plan_id": plan.plan_id,
                "plan_name": (plan.name or "")[:255],
                "dimension_key": f.dimension_key,
                "dimension_label": f.dimension_label,
                "value_text": (f.value_text or "")[:2048],
                "normalized_value": (f.normalized_value or "")[:2048],
                "unit": f.unit or "",
                "condition_text": (f.condition_text or "")[:1024],
                "applicability": f.applicability or "",
                "source_chunk_id": f.source_chunk_id or "",
                "source_page": f.source_page or -1,
                "source_section": (f.source_section or "")[:512],
                "confidence": float(f.confidence),
                "created_at": self.milvus.now_ts(),
                "dense_embedding": chunk_embedding_map.get(f.source_chunk_id or "", [0.0] * target_dim),
                "sparse_embedding": chunk_sparse_map.get(f.source_chunk_id or "", {}),
            }
            for f in facts
        ]

        chunk_count = 0
        fact_count = 0
        chunk_err: Exception | None = None
        fact_err: Exception | None = None

        try:
            chunk_count = self.milvus.upsert_chunks(chunk_records)
        except Exception as exc:
            chunk_err = exc
            self.logger.warning("ingest.milvus.chunk_sync_failed plan=%s err=%s", plan.name, exc)

        try:
            fact_count = self.milvus.upsert_facts(fact_records)
        except Exception as exc:
            fact_err = exc
            self.logger.warning("ingest.milvus.fact_sync_failed plan=%s err=%s", plan.name, exc)

        if chunk_err or fact_err:
            self.logger.warning(
                "ingest.milvus.partial_sync plan=%s chunk_ok=%d/%d fact_ok=%d/%d",
                plan.name,
                chunk_count,
                len(chunk_records),
                fact_count,
                len(fact_records),
            )
