from __future__ import annotations

import json
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, engine, get_db, session_scope
from app.models import Plan
from app.schemas import (
    ChatMessageRequest,
    ChatMessageResponse,
    CompareRequest,
    CompareResponse,
    CreateSessionRequest,
    DimensionOut,
    IngestResponse,
    PlanOut,
    SessionStateOut,
)
from app.services.chat import ChatService
from app.services.compare import CompareService
from app.services.dimensions import all_dimensions
from app.services.ingestion import IngestionService


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

compare_service = CompareService()
chat_service = ChatService()


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService()


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ingest/run", response_model=IngestResponse)
def run_ingestion(db: Session = Depends(get_db)) -> IngestResponse:
    ingestion_service = get_ingestion_service()
    plans_processed, chunks_written, facts_written = ingestion_service.ingest_all(db)
    return IngestResponse(
        plans_processed=plans_processed,
        chunks_written=chunks_written,
        facts_written=facts_written,
    )


@app.get("/api/plans", response_model=list[PlanOut])
def list_plans(db: Session = Depends(get_db)) -> list[PlanOut]:
    plans = db.execute(select(Plan).order_by(Plan.name.asc())).scalars().all()
    return [PlanOut(plan_id=p.plan_id, name=p.name, source_file=p.source_file) for p in plans]


@app.get("/api/dimensions", response_model=list[DimensionOut])
def list_dimensions() -> list[DimensionOut]:
    return [DimensionOut(**d) for d in all_dimensions()]


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest, db: Session = Depends(get_db)) -> CompareResponse:
    if len(req.plan_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two plans are required for comparison")
    return compare_service.build_compare(db, req.plan_ids, req.dimensions, req.filters)


@app.post("/api/chat/session", response_model=SessionStateOut)
def create_chat_session(req: CreateSessionRequest, db: Session = Depends(get_db)) -> SessionStateOut:
    return chat_service.create_session(db, req.user_id)


@app.get("/api/chat/session/{session_id}", response_model=SessionStateOut)
def get_chat_state(session_id: str, db: Session = Depends(get_db)) -> SessionStateOut:
    return chat_service.get_state(db, session_id)


@app.post("/api/chat/message", response_model=ChatMessageResponse)
def chat_message(req: ChatMessageRequest, db: Session = Depends(get_db)) -> ChatMessageResponse:
    return chat_service.post_message(
        db,
        req.session_id,
        req.content,
        selected_plans=req.selected_plans,
        dimensions=req.dimensions,
    )


@app.post("/api/chat/message/stream")
def chat_message_stream(req: ChatMessageRequest) -> StreamingResponse:
    def event_gen():
        yield f"event: token\ndata: {json.dumps({'text': '正在分析中，请稍候...\\n'}, ensure_ascii=False)}\n\n"
        try:
            with session_scope() as db:
                result = chat_service.post_message(
                    db,
                    req.session_id,
                    req.content,
                    selected_plans=req.selected_plans,
                    dimensions=req.dimensions,
                )
            text = result.reply or ""
            step = 24
            for i in range(0, len(text), step):
                piece = text[i : i + step]
                yield f"event: token\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {json.dumps(result.model_dump(mode='json'), ensure_ascii=False)}\n\n"
        except Exception as exc:
            payload = {"error": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
