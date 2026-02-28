from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _uuid() -> str:
    return uuid4().hex


class Plan(Base):
    __tablename__ = "plans"

    plan_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    product_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    product_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    chunks: Mapped[list[PolicyChunk]] = relationship(back_populates="plan", cascade="all, delete-orphan")
    facts: Mapped[list[PolicyFact]] = relationship(back_populates="plan", cascade="all, delete-orphan")


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(String(64), ForeignKey("plans.plan_id"), index=True)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    paragraph_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bbox_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    plan: Mapped[Plan] = relationship(back_populates="chunks")
    facts: Mapped[list[PolicyFact]] = relationship(back_populates="source_chunk")


class PolicyFact(Base):
    __tablename__ = "policy_facts"

    fact_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(String(64), ForeignKey("plans.plan_id"), index=True)
    dimension_key: Mapped[str] = mapped_column(String(128), index=True)
    dimension_label: Mapped[str] = mapped_column(String(255), nullable=False)
    value_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    condition_text: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    applicability: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_chunk_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("policy_chunks.chunk_id"), nullable=True)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_section: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    plan: Mapped[Plan] = relationship(back_populates="facts")
    source_chunk: Mapped[PolicyChunk | None] = relationship(back_populates="facts")


Index("idx_policy_fact_plan_dim", PolicyFact.plan_id, PolicyFact.dimension_key)
Index("idx_policy_fact_dim_numeric", PolicyFact.dimension_key, PolicyFact.numeric_value)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    turns: Mapped[list[ChatTurn]] = relationship(back_populates="session", cascade="all, delete-orphan")
    state: Mapped[SessionState | None] = relationship(back_populates="session", uselist=False, cascade="all, delete-orphan")


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    turn_id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("chat_sessions.session_id"), index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="turns")


class SessionState(Base):
    __tablename__ = "session_state"

    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("chat_sessions.session_id"), primary_key=True)
    selected_plans: Mapped[list[str]] = mapped_column(JSON, default=list)
    dimensions: Mapped[list[str]] = mapped_column(JSON, default=list)
    filters: Mapped[dict] = mapped_column(JSON, default=dict)
    last_table_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="state")
