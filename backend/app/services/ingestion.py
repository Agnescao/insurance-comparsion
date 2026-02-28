from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Plan, PolicyChunk, PolicyFact
from app.services.chunking import ChunkDoc, HybridChunker
from app.services.embeddings import build_embedding_provider
from app.services.fact_extractor import FactExtractor, FactRecord
from app.services.milvus_store import MilvusStore
from app.services.parser import PDFParser, ParsedPolicy


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


class IngestionService:
    def __init__(self) -> None:
        self.embedding_provider = build_embedding_provider()
        self.parser = PDFParser()
        self.chunker = HybridChunker(self.embedding_provider)
        self.fact_extractor = FactExtractor()
        self.milvus = MilvusStore()
        self.milvus.connect()
        self.milvus.ensure_collections()

    def ingest_all(self, db: Session) -> tuple[int, int, int]:
        pdf_paths = sorted(Path(settings.data_dir).glob("*.pdf"))
        plans_processed = 0
        chunks_written = 0
        facts_written = 0

        for pdf_path in pdf_paths:
            p_count, c_count, f_count = self.ingest_one(db, pdf_path)
            plans_processed += p_count
            chunks_written += c_count
            facts_written += f_count

        return plans_processed, chunks_written, facts_written

    def ingest_one(self, db: Session, pdf_path: Path) -> tuple[int, int, int]:
        parsed = self.parser.parse(pdf_path)

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

        chunk_docs = self.chunker.chunk_policy(parsed)
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

        facts = self.fact_extractor.extract_from_chunks(chunk_docs)
        fact_models: list[PolicyFact] = []
        chunk_by_page: dict[int, PolicyChunk] = {c.page_start or -1: c for c in chunks}
        for fact in facts:
            source_chunk = chunk_by_page.get(fact.source_page or -1)
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
                metadata_json={"source_file": str(pdf_path)},
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
            }
            for f in facts
        ]
        (doc_dir / "facts.json").write_text(json.dumps(facts_json, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sync_milvus(self, plan: Plan, chunks: list[PolicyChunk], facts: list[PolicyFact]) -> None:
        if not self.milvus.enabled:
            return

        chunk_embedding_map: dict[str, list[float]] = {
            c.chunk_id: (c.embedding or [0.0] * settings.embedding_dim) for c in chunks
        }

        chunk_records = [
            {
                "chunk_id": c.chunk_id,
                "plan_id": plan.plan_id,
                "plan_name": plan.name,
                "product_version": plan.product_version or "unknown",
                "section_path": c.section_path or "",
                "page_start": c.page_start or -1,
                "page_end": c.page_end or -1,
                "source_ref": f"{plan.source_file}#p{c.page_start or 0}",
                "language": plan.language or "zh",
                "text": c.text[:8192],
                "created_at": self.milvus.now_ts(),
                "embedding": chunk_embedding_map[c.chunk_id],
            }
            for c in chunks
        ]

        fact_records = [
            {
                "fact_id": f.fact_id,
                "plan_id": plan.plan_id,
                "dimension_key": f.dimension_key,
                "dimension_label": f.dimension_label,
                "value_text": (f.value_text or "")[:2048],
                "normalized_value": (f.normalized_value or "")[:1024],
                "unit": f.unit or "",
                "condition_text": (f.condition_text or "")[:1024],
                "applicability": f.applicability or "",
                "source_chunk_id": f.source_chunk_id or "",
                "source_page": f.source_page or -1,
                "source_section": (f.source_section or "")[:512],
                "confidence": float(f.confidence),
                "created_at": self.milvus.now_ts(),
                "embedding": chunk_embedding_map.get(f.source_chunk_id or "", [0.0] * settings.embedding_dim),
            }
            for f in facts
        ]

        self.milvus.upsert_chunks(chunk_records)
        self.milvus.upsert_facts(fact_records)
