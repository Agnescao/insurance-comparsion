from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ChatSession, ChatTurn, Plan, SessionState
from app.schemas import ChatMessageResponse, ChatTurnOut, CompareResponse, SessionStateOut
from app.services.compare import CompareService
from app.services.dimensions import DEFAULT_DIMENSIONS, detect_dimensions


class ChatService:
    def __init__(self) -> None:
        self.compare_service = CompareService()

    def create_session(self, db: Session, user_id: str | None = None) -> SessionStateOut:
        session = ChatSession(user_id=user_id)
        db.add(session)
        db.flush()

        all_plan_ids = [p.plan_id for p in db.execute(select(Plan).order_by(Plan.name)).scalars().all()]
        state = SessionState(
            session_id=session.session_id,
            selected_plans=all_plan_ids[:2],
            dimensions=DEFAULT_DIMENSIONS[:6],
            filters={},
        )
        db.add(state)
        db.flush()
        return self._state_out(state)

    def get_or_create_state(self, db: Session, session_id: str) -> SessionState:
        state = db.get(SessionState, session_id)
        if state:
            return state

        state = SessionState(
            session_id=session_id,
            selected_plans=[],
            dimensions=DEFAULT_DIMENSIONS[:6],
            filters={},
        )
        db.add(state)
        db.flush()
        return state

    def get_state(self, db: Session, session_id: str) -> SessionStateOut:
        state = self.get_or_create_state(db, session_id)
        return self._state_out(state)

    def post_message(self, db: Session, session_id: str, content: str) -> ChatMessageResponse:
        session = db.get(ChatSession, session_id)
        if not session:
            session = ChatSession(session_id=session_id)
            db.add(session)
            db.flush()

        state = self.get_or_create_state(db, session_id)

        user_turn = ChatTurn(session_id=session_id, role="user", content=content)
        db.add(user_turn)
        db.flush()

        added_dims = self._update_dimensions(state, content)
        added_plans = self._update_plans(db, state, content)

        compare: CompareResponse | None = None
        if len(state.selected_plans) >= 2:
            compare = self.compare_service.build_compare(
                db,
                plan_ids=state.selected_plans,
                dimensions=state.dimensions,
                filters=state.filters,
            )
            state.last_table_snapshot = compare.model_dump(mode="json")

        reply = self._build_reply(added_plans, added_dims, state, compare)
        assistant_turn = ChatTurn(session_id=session_id, role="assistant", content=reply)
        db.add(assistant_turn)
        db.flush()

        turns = (
            db.execute(select(ChatTurn).where(ChatTurn.session_id == session_id).order_by(ChatTurn.timestamp.asc()))
            .scalars()
            .all()
        )

        return ChatMessageResponse(
            session_id=session_id,
            reply=reply,
            state=self._state_out(state),
            compare=compare,
            turns=[ChatTurnOut(role=t.role, content=t.content, timestamp=t.timestamp) for t in turns[-30:]],
        )

    def _update_dimensions(self, state: SessionState, content: str) -> list[str]:
        detected = detect_dimensions(content)
        added: list[str] = []
        if not state.dimensions:
            state.dimensions = DEFAULT_DIMENSIONS[:6]
        for d in detected:
            if d not in state.dimensions:
                state.dimensions.append(d)
                added.append(d)
        return added

    def _update_plans(self, db: Session, state: SessionState, content: str) -> list[str]:
        plans = db.execute(select(Plan).order_by(Plan.name)).scalars().all()
        lowered = content.lower()
        added: list[str] = []

        for p in plans:
            name = p.name.lower()
            file_base = p.source_file.lower().split("\\")[-1]
            if name in lowered or file_base in lowered:
                if p.plan_id not in state.selected_plans:
                    state.selected_plans.append(p.plan_id)
                    added.append(p.name)

        # Ensure at least two plans are selected for compare table
        if len(state.selected_plans) < 2:
            for p in plans:
                if p.plan_id not in state.selected_plans:
                    state.selected_plans.append(p.plan_id)
                if len(state.selected_plans) >= 2:
                    break

        return added

    def _build_reply(
        self,
        added_plans: list[str],
        added_dims: list[str],
        state: SessionState,
        compare: CompareResponse | None,
    ) -> str:
        lines: list[str] = []
        if added_plans:
            lines.append(f"已加入计划: {', '.join(added_plans)}")
        if added_dims:
            lines.append(f"已加入维度: {', '.join(added_dims)}")

        if compare and compare.rows:
            diff_count = sum(1 for r in compare.rows if r.is_different)
            lines.append(f"当前比较包含 {len(compare.plan_ids)} 个计划、{len(compare.rows)} 个维度，其中 {diff_count} 个维度存在明显差异。")
            if not added_plans and not added_dims:
                lines.append("我已基于当前上下文刷新对比表，你可以继续提问“增加某个维度”或“加入某个计划”。")
        else:
            lines.append("请至少选择两个计划后再进行对比。")

        return "\n".join(lines)

    def _state_out(self, state: SessionState) -> SessionStateOut:
        return SessionStateOut(
            session_id=state.session_id,
            selected_plans=state.selected_plans or [],
            dimensions=state.dimensions or [],
            filters=state.filters or {},
            updated_at=state.updated_at,
        )
