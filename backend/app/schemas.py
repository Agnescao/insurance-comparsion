from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PlanOut(BaseModel):
    plan_id: str
    name: str
    source_file: str


class DimensionOut(BaseModel):
    key: str
    label: str


class CompareRequest(BaseModel):
    plan_ids: list[str] = Field(min_length=1)
    dimensions: list[str] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)


class SourceRef(BaseModel):
    page: int | None = None
    section: str | None = None
    quote: str | None = None


class CellValue(BaseModel):
    value: str
    confidence: float
    source: SourceRef


class CompareRow(BaseModel):
    dimension_key: str
    dimension_label: str
    is_different: bool
    plan_values: dict[str, CellValue]


class CompareResponse(BaseModel):
    generated_at: datetime
    plan_ids: list[str]
    dimensions: list[str]
    rows: list[CompareRow]


class IngestResponse(BaseModel):
    plans_processed: int
    chunks_written: int
    facts_written: int


class CreateSessionRequest(BaseModel):
    user_id: str | None = None


class SessionStateOut(BaseModel):
    session_id: str
    selected_plans: list[str]
    dimensions: list[str]
    filters: dict
    updated_at: datetime | None = None


class ChatMessageRequest(BaseModel):
    session_id: str
    content: str
    selected_plans: list[str] | None = None
    dimensions: list[str] | None = None


class ChatTurnOut(BaseModel):
    role: str
    content: str
    timestamp: datetime


class ChatMessageResponse(BaseModel):
    session_id: str
    reply: str
    state: SessionStateOut
    compare: CompareResponse | None = None
    turns: list[ChatTurnOut] = Field(default_factory=list)
