from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ChatSession, ChatTurn, Plan, PolicyChunk, PolicyFact, SessionState
from app.schemas import (
    CellValue,
    ChatMessageResponse,
    ChatTurnOut,
    CompareResponse,
    CompareRow,
    SessionStateOut,
    SourceRef,
)
from app.services.compare import CompareService
from app.services.dimensions import CONDITION_TERMS, condition_dimension_for_query, detect_dimensions
from app.services.hybrid_retriever import HybridRetriever
from app.services.llm_planner import LLMPlanner, PlannerResult


@dataclass
class AutoDiscoveryOutcome:
    triggered: bool = False
    status: str = ""
    message: str = ""
    selected_plan_ids: list[str] = field(default_factory=list)
    evidence_by_plan: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class ChatService:
    def __init__(self) -> None:
        self.compare_service = CompareService()
        self.planner = LLMPlanner()
        self.retriever = HybridRetriever()

    def create_session(self, db: Session, user_id: str | None = None) -> SessionStateOut:
        session = ChatSession(user_id=user_id)
        db.add(session)
        db.flush()

        state = SessionState(
            session_id=session.session_id,
            selected_plans=[],
            dimensions=[],
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
            dimensions=[],
            filters={},
        )
        db.add(state)
        db.flush()
        return state

    def get_state(self, db: Session, session_id: str) -> SessionStateOut:
        return self._state_out(self.get_or_create_state(db, session_id))

    def post_message(
        self,
        db: Session,
        session_id: str,
        content: str,
        selected_plans: list[str] | None = None,
        dimensions: list[str] | None = None,
    ) -> ChatMessageResponse:
        session = db.get(ChatSession, session_id)
        if not session:
            session = ChatSession(session_id=session_id)
            db.add(session)
            db.flush()

        state = self.get_or_create_state(db, session_id)
        plans = db.execute(select(Plan).order_by(Plan.name)).scalars().all()
        plan_name_map = {p.plan_id: p.name for p in plans}
        valid_plan_ids = {p.plan_id for p in plans}

        if selected_plans is not None:
            sanitized_plan_ids = []
            for pid in selected_plans:
                if isinstance(pid, str) and pid in valid_plan_ids and pid not in sanitized_plan_ids:
                    sanitized_plan_ids.append(pid)
            state.selected_plans = sanitized_plan_ids
        if dimensions is not None:
            sanitized_dims = []
            for dim in dimensions:
                if isinstance(dim, str) and dim not in sanitized_dims:
                    sanitized_dims.append(dim)
            state.dimensions = sanitized_dims
        allow_plan_expansion = self._should_expand_plan_scope(content, state, plans)

        db.add(ChatTurn(session_id=session_id, role="user", content=content))
        db.flush()

        planner_result = self._plan_actions(content, state, plans)
        added_dims, added_plans = self._apply_actions(
            db,
            state,
            plans,
            planner_result,
            content,
            allow_plan_expansion,
        )
        retrieval_keywords = self.planner.extract_retrieval_keywords(
            query=content,
            state={
                "selected_plans": state.selected_plans or [],
                "dimensions": state.dimensions or [],
                "filters": state.filters or {},
            },
        )
        retrieval_query = " ".join([kw for kw in retrieval_keywords if kw]).strip() or content
        auto_discovery = self._auto_route_by_keyword_retrieval(
            db=db,
            state=state,
            plans=plans,
            query=content,
            retrieval_query=retrieval_query,
            plan_name_map=plan_name_map,
        )
        enforce_min_plans = not (auto_discovery.triggered and auto_discovery.status in {"none", "single"})
        self._ensure_min_plans(state, plans, enforce=enforce_min_plans)
        evidence_by_plan = auto_discovery.evidence_by_plan or self.retriever.retrieve_plan_evidence(
            db,
            query=retrieval_query,
            plan_ids=state.selected_plans or [],
            per_plan_k=2,
        )
        compare = self._build_compare_if_possible(
            db,
            state,
            content,
            evidence_by_plan=evidence_by_plan,
        )

        deterministic = self._build_reply(
            added_plans=added_plans,
            added_dims=added_dims,
            compare=compare,
            query=content,
            plan_name_map=plan_name_map,
            evidence_by_plan=evidence_by_plan,
            selected_plan_ids=state.selected_plans or [],
            auto_discovery_status=auto_discovery.status,
            auto_discovery_message=auto_discovery.message,
        )
        llm_reply = ""
        has_retrieval_hits = any(evidence_by_plan.get(pid) for pid in (state.selected_plans or []))
        if has_retrieval_hits:
            llm_reply = self._maybe_llm_reply(content, state, compare, plan_name_map, evidence_by_plan)
        reply = llm_reply or deterministic
        reply = self._replace_plan_ids_with_names(reply, plan_name_map)
        if self._looks_insufficient(reply):
            reply = deterministic

        db.add(ChatTurn(session_id=session_id, role="assistant", content=reply))
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

    def _plan_actions(
        self,
        content: str,
        state: SessionState,
        plans: list[Plan],
    ) -> PlannerResult:
        if self._is_simple_context_query(content, state):
            dims = detect_dimensions(content)
            actions: list[dict] = []
            if dims:
                actions.append({"type": "add_dimensions", "dimensions": dims})
            actions.append({"type": "refresh_compare"})
            return PlannerResult(mode="context_compare", actions=actions, reasoning="fast_rule_path")

        return self.planner.plan(
            query=content,
            state={
                "selected_plans": state.selected_plans or [],
                "dimensions": state.dimensions or [],
                "filters": state.filters or {},
            },
            available_plans=[{"plan_id": p.plan_id, "name": p.name, "source_file": p.source_file} for p in plans],
        )

    def _apply_actions(
        self,
        db: Session,
        state: SessionState,
        plans: list[Plan],
        planner_result: PlannerResult,
        content: str,
        allow_plan_expansion: bool,
    ) -> tuple[list[str], list[str]]:
        added_dims: list[str] = []
        added_plans: list[str] = []

        for action in planner_result.actions:
            action_type = str(action.get("type", "")).strip().lower()
            if action_type == "add_dimensions":
                added_dims.extend(self._apply_add_dimensions(state, action.get("dimensions", [])))
            elif action_type == "set_filters":
                filters = action.get("filters", {})
                if isinstance(filters, dict):
                    state.filters = {**(state.filters or {}), **filters}
            elif action_type == "set_compare_plans":
                plan_ids = action.get("plan_ids", [])
                if isinstance(plan_ids, list):
                    requested = [pid for pid in plan_ids if isinstance(pid, str)]
                    if allow_plan_expansion:
                        state.selected_plans = requested
                    else:
                        allowed = set(state.selected_plans or [])
                        narrowed = [pid for pid in requested if pid in allowed]
                        if len(narrowed) >= 2:
                            state.selected_plans = narrowed
            elif action_type == "discover_products":
                query = str(action.get("query") or content)
                top_k = int(action.get("top_k", 3) or 3)
                candidate_ids = None if allow_plan_expansion else (state.selected_plans or None)
                discovered = self.retriever.discover_plan_ids(
                    db,
                    query=query,
                    top_k=top_k,
                    candidate_plan_ids=candidate_ids,
                )
                if allow_plan_expansion:
                    newly_added: list[str] = []
                    for pid in discovered:
                        if pid not in (state.selected_plans or []):
                            if not state.selected_plans:
                                state.selected_plans = []
                            state.selected_plans.append(pid)
                            newly_added.append(pid)
                    if newly_added:
                        added_plans.extend(self._plan_names_by_ids(plans, newly_added))
                elif discovered:
                    state.selected_plans = discovered

        if allow_plan_expansion and not added_plans:
            added_plans.extend(self._update_plans_by_text(plans, state, content))
        if not added_dims:
            added_dims.extend(self._update_dimensions_fallback(state, content))

        c_key = condition_dimension_for_query(content)
        if c_key and c_key not in (state.dimensions or []):
            if not state.dimensions:
                state.dimensions = []
            state.dimensions.append(c_key)
            added_dims.append(c_key)

        return list(dict.fromkeys(added_dims)), list(dict.fromkeys(added_plans))

    def _ensure_min_plans(self, state: SessionState, plans: list[Plan], *, enforce: bool = True) -> None:
        if not enforce:
            return
        if len(state.selected_plans or []) >= 2:
            return
        if not state.selected_plans:
            state.selected_plans = []
        for p in plans:
            if p.plan_id not in state.selected_plans:
                state.selected_plans.append(p.plan_id)
            if len(state.selected_plans) >= 2:
                break

    def _build_compare_if_possible(
        self,
        db: Session,
        state: SessionState,
        content: str,
        evidence_by_plan: dict[str, list[dict[str, Any]]] | None = None,
    ) -> CompareResponse | None:
        if len(state.selected_plans or []) < 2:
            return None
        if not (state.dimensions or []):
            inferred_dims = detect_dimensions(content)
            if not inferred_dims:
                lowered = (content or "").lower()
                if any(tok in lowered for tok in ("\u624b\u672f", "\u624b\u8853", "surgery", "operation")):
                    inferred_dims = ["coverage_surgery"]
                else:
                    inferred_dims = ["coverage_hospitalization"]
            state.dimensions = list(dict.fromkeys(inferred_dims))
        compare = self.compare_service.build_compare(
            db,
            plan_ids=state.selected_plans,
            dimensions=state.dimensions,
            filters=state.filters,
        )
        local_evidence_by_plan = evidence_by_plan or {}
        if condition_dimension_for_query(content) and not local_evidence_by_plan:
            evidence_by_plan = self.retriever.retrieve_plan_evidence(
                db,
                query=content,
                plan_ids=state.selected_plans or [],
                per_plan_k=2,
            )
            local_evidence_by_plan = evidence_by_plan
        self._backfill_condition_row_from_chunks(
            db,
            compare,
            state.selected_plans,
            content,
            evidence_by_plan=local_evidence_by_plan,
        )
        state.last_table_snapshot = compare.model_dump(mode="json")
        return compare

    def _maybe_llm_reply(
        self,
        content: str,
        state: SessionState,
        compare: CompareResponse | None,
        plan_name_map: dict[str, str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
    ) -> str:
        del compare  # retrieval-grounded summary no longer depends on table availability
        if not (state.selected_plans or []):
            return ""
        payload = self._compact_evidence_for_llm(state.selected_plans or [], evidence_by_plan, plan_name_map)
        if not payload or not any(item.get("evidence") for item in payload):
            return ""
        plan_names = [plan_name_map.get(pid, pid) for pid in (state.selected_plans or [])]
        return self.planner.summarize_evidence_compare(
            query=content,
            plan_names=plan_names,
            evidence_payload=payload,
            state_payload={
                "selected_plans": state.selected_plans,
                "dimensions": state.dimensions,
                "filters": state.filters,
            },
        )

    def _is_simple_context_query(self, content: str, state: SessionState) -> bool:
        text = (content or "").strip().lower()
        if not text or len(text) > 90:
            return False
        if len(state.selected_plans or []) < 2:
            return False
        discovery_keywords = (
            "\u63a8\u8350",
            "\u627e\u8ba1\u5212",
            "\u66f4\u5339\u914d",
            "best plan",
            "find plan",
            "recommend",
        )
        if any(k in text for k in discovery_keywords):
            return False
        return True

    def _should_expand_plan_scope(self, content: str, state: SessionState, plans: list[Plan]) -> bool:
        text = self._normalize_for_match(content)
        if not text:
            return False

        explicit_expand_keywords = (
            "\u63a8\u8350\u5176\u4ed6\u8ba1\u5212",
            "\u63a8\u8350\u65b0\u8ba1\u5212",
            "\u627e\u65b0\u8ba1\u5212",
            "\u627e\u66f4\u591a\u8ba1\u5212",
            "\u52a0\u5165\u8ba1\u5212",
            "\u6dfb\u52a0\u8ba1\u5212",
            "\u6269\u5c55\u5bf9\u6bd4",
            "recommend another plan",
            "recommend other plans",
            "find a new plan",
            "add this plan",
            "include this plan",
        )
        if any(self._normalize_for_match(k) in text for k in explicit_expand_keywords):
            return True

        selected = set(state.selected_plans or [])
        for p in plans:
            if p.plan_id in selected:
                continue
            plan_name = self._normalize_for_match(p.name or "")
            source = self._normalize_for_match((p.source_file or "").replace("\\", "/").split("/")[-1])
            if plan_name and plan_name in text:
                return True
            if source and source in text:
                return True
        return False

    def _normalize_for_match(self, text: str) -> str:
        lowered = (text or "").strip().lower()
        return re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)

    def _apply_add_dimensions(self, state: SessionState, dimensions: list[str]) -> list[str]:
        if not state.dimensions:
            state.dimensions = []
        added: list[str] = []
        for dim in dimensions:
            if isinstance(dim, str) and dim not in state.dimensions:
                state.dimensions.append(dim)
                added.append(dim)
        return added

    def _update_dimensions_fallback(self, state: SessionState, content: str) -> list[str]:
        return self._apply_add_dimensions(state, detect_dimensions(content))

    def _update_plans_by_text(self, plans: list[Plan], state: SessionState, content: str) -> list[str]:
        lowered = (content or "").lower()
        if not state.selected_plans:
            state.selected_plans = []
        added: list[str] = []
        for p in plans:
            name = (p.name or "").lower()
            source = (p.source_file or "").lower().replace("\\", "/").split("/")[-1]
            if name in lowered or source in lowered:
                if p.plan_id not in state.selected_plans:
                    state.selected_plans.append(p.plan_id)
                    added.append(p.name)
        return added

    def _plan_names_by_ids(self, plans: list[Plan], ids: list[str]) -> list[str]:
        by_id = {p.plan_id: p.name for p in plans}
        return [by_id[i] for i in ids if i in by_id]

    def _auto_route_by_keyword_retrieval(
        self,
        db: Session,
        state: SessionState,
        plans: list[Plan],
        query: str,
        retrieval_query: str,
        plan_name_map: dict[str, str],
    ) -> AutoDiscoveryOutcome:
        selected = list(state.selected_plans or [])
        fallback_evidence = {pid: [] for pid in selected}
        semantic_query = (retrieval_query or query or "").strip()
        if not semantic_query:
            return AutoDiscoveryOutcome(evidence_by_plan=fallback_evidence, selected_plan_ids=selected)

        if selected:
            selected_evidence = self.retriever.retrieve_plan_evidence(
                db,
                query=semantic_query,
                plan_ids=selected,
                per_plan_k=3,
            )
            selected_covered = self._covered_plan_ids_from_evidence(
                query=query,
                plan_ids=selected,
                evidence_by_plan=selected_evidence,
                plan_name_map=plan_name_map,
            )
            if selected_covered:
                return AutoDiscoveryOutcome(
                    triggered=False,
                    status="selected",
                    selected_plan_ids=selected,
                    evidence_by_plan=selected_evidence,
                )

        if selected:
            candidate_ids = [p.plan_id for p in plans if p.plan_id not in selected]
        else:
            candidate_ids = [p.plan_id for p in plans]

        if not candidate_ids:
            return AutoDiscoveryOutcome(
                triggered=True,
                status="none",
                message="\u5f53\u524d\u5df2\u9009\u8ba1\u5212\u672a\u68c0\u7d22\u5230\u76f8\u5173\u5185\u5bb9\uff0c\u4e14\u6ca1\u6709\u66f4\u591a\u53ef\u68c0\u7d22\u8ba1\u5212\u3002",
                selected_plan_ids=selected,
                evidence_by_plan=fallback_evidence,
            )

        discovered = self.retriever.discover_plan_ids(
            db,
            query=semantic_query,
            top_k=max(4, min(8, len(candidate_ids))),
            candidate_plan_ids=candidate_ids,
        )
        discovered = [pid for pid in discovered if pid in candidate_ids]
        if not discovered:
            return AutoDiscoveryOutcome(
                triggered=True,
                status="none",
                message="\u5f53\u524d\u5df2\u9009\u8ba1\u5212\u672a\u68c0\u7d22\u5230\u76f8\u5173\u5185\u5bb9\uff0c\u4e14\u672a\u5728\u5176\u4ed6\u8ba1\u5212\u4e2d\u627e\u5230\u76f8\u5173\u7ed3\u679c\u3002",
                selected_plan_ids=selected,
                evidence_by_plan=fallback_evidence,
            )

        selected_new = list(dict.fromkeys(discovered[:4]))
        discovered_evidence = self.retriever.retrieve_plan_evidence(
            db,
            query=semantic_query,
            plan_ids=selected_new,
            per_plan_k=3,
        )
        discovered_covered = self._covered_plan_ids_from_evidence(
            query=query,
            plan_ids=selected_new,
            evidence_by_plan=discovered_evidence,
            plan_name_map=plan_name_map,
        )
        if not discovered_covered and not any(discovered_evidence.values()):
            return AutoDiscoveryOutcome(
                triggered=True,
                status="none",
                message="\u5df2\u5c1d\u8bd5\u68c0\u7d22\u65b0\u8ba1\u5212\uff0c\u4f46\u6682\u672a\u627e\u5230\u76f4\u63a5\u76f8\u5173\u7684\u6761\u6b3e\u8bc1\u636e\u3002",
                selected_plan_ids=selected_new,
                evidence_by_plan=discovered_evidence,
            )

        matched = discovered_covered or [pid for pid in selected_new if discovered_evidence.get(pid)]
        if not matched:
            return AutoDiscoveryOutcome(
                triggered=True,
                status="none",
                message="\u5df2\u5c1d\u8bd5\u68c0\u7d22\u65b0\u8ba1\u5212\uff0c\u4f46\u6682\u672a\u627e\u5230\u76f4\u63a5\u76f8\u5173\u7684\u6761\u6b3e\u8bc1\u636e\u3002",
                selected_plan_ids=selected_new,
                evidence_by_plan=discovered_evidence,
            )

        selected_matched = list(dict.fromkeys(matched[:4]))
        state.selected_plans = selected_matched
        names = [plan_name_map.get(pid, pid) for pid in selected_matched]
        if len(selected_matched) == 1:
            message = f"\u5f53\u524d\u5df2\u9009\u8ba1\u5212\u672a\u547d\u4e2d\uff0c\u5df2\u5207\u6362\u5230\u66f4\u76f8\u5173\u7684\u65b0\u8ba1\u5212: {names[0]}\u3002"
            status = "single"
        else:
            message = (
                "\u5f53\u524d\u5df2\u9009\u8ba1\u5212\u672a\u547d\u4e2d\uff0c"
                f"\u5df2\u5207\u6362\u5230\u66f4\u76f8\u5173\u7684\u8ba1\u5212: {', '.join(names)}\u3002"
            )
            status = "multi"
        return AutoDiscoveryOutcome(
            triggered=True,
            status=status,
            message=message,
            selected_plan_ids=selected_matched,
            evidence_by_plan={pid: discovered_evidence.get(pid, []) for pid in selected_matched},
        )

    def _covered_plan_ids_from_evidence(
        self,
        *,
        query: str,
        plan_ids: list[str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
        plan_name_map: dict[str, str],
    ) -> list[str]:
        valid_ids = [pid for pid in plan_ids if isinstance(pid, str) and pid]
        if not valid_ids:
            return []
        if not any(evidence_by_plan.get(pid) for pid in valid_ids):
            return []

        plans_payload = [{"plan_id": pid, "name": plan_name_map.get(pid, pid)} for pid in valid_ids]
        covered = self.planner.assess_query_coverage(
            query=query,
            plans=plans_payload,
            evidence_by_plan={pid: evidence_by_plan.get(pid, []) for pid in valid_ids},
        )
        if covered is None:
            return [pid for pid in valid_ids if evidence_by_plan.get(pid)]
        covered_set = set(covered)
        return [pid for pid in valid_ids if pid in covered_set]

    def _backfill_condition_row_from_chunks(
        self,
        db: Session,
        compare: CompareResponse,
        selected_plan_ids: list[str],
        query: str,
        evidence_by_plan: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        c_key = condition_dimension_for_query(query)
        if not c_key:
            return
        terms = CONDITION_TERMS.get(c_key, ())
        if not terms:
            return

        row = next((r for r in compare.rows if r.dimension_key == c_key), None)
        if row is None:
            row = CompareRow(
                dimension_key=c_key,
                dimension_label=f"\u75be\u75c5\u573a\u666f: {c_key.removeprefix('condition_').replace('_', ' ')}",
                is_different=False,
                plan_values={},
            )
            compare.rows.append(row)
            if c_key not in compare.dimensions:
                compare.dimensions.append(c_key)

        evidence_by_plan = evidence_by_plan or {}
        for pid in selected_plan_ids:
            current = row.plan_values.get(pid)
            if current and current.value and current.value != "\u672a\u63d0\u53d6\u5230":
                continue
            matched = self._find_condition_fact_evidence(db, pid, terms)
            if not matched:
                matched = self._find_condition_chunk_evidence(db, pid, terms)
            if not matched:
                matched = self._find_condition_evidence_from_retrieval(evidence_by_plan.get(pid, []), terms)
            if not matched:
                if pid not in row.plan_values:
                    row.plan_values[pid] = CellValue(
                        value="\u672a\u63d0\u53d6\u5230",
                        confidence=0.0,
                        source=SourceRef(page=None, section=None, quote=None),
                    )
                continue
            row.plan_values[pid] = CellValue(
                value=matched["value"],
                confidence=float(matched.get("confidence") or 0.56),
                source=SourceRef(page=matched["page"], section=matched["section"], quote=matched["quote"]),
            )

        non_empty = [
            (v.value or "").strip().lower()
            for v in row.plan_values.values()
            if (v.value or "").strip() and (v.value or "").strip() != "\u672a\u63d0\u53d6\u5230"
        ]
        row.is_different = len(set(non_empty)) > 1 if non_empty else False

    def _find_condition_fact_evidence(self, db: Session, plan_id: str, terms: tuple[str, ...]) -> dict | None:
        facts = (
            db.execute(
                select(PolicyFact)
                .where(PolicyFact.plan_id == plan_id)
                .order_by(PolicyFact.confidence.desc(), PolicyFact.created_at.desc())
            )
            .scalars()
            .all()
        )
        lower_terms = [t.lower() for t in terms if t]
        for fact in facts:
            candidates = [
                fact.dimension_key or "",
                fact.dimension_label or "",
                fact.condition_text or "",
                fact.value_text or "",
                fact.normalized_value or "",
                fact.source_quote or "",
            ]
            merged = " ".join(candidates).lower()
            if any(t in merged for t in lower_terms):
                value = self._shorten((fact.value_text or fact.source_quote or fact.condition_text or "").strip(), 180)
                if not value:
                    continue
                return {
                    "value": value,
                    "page": fact.source_page,
                    "section": fact.source_section,
                    "quote": self._shorten((fact.source_quote or value), 300),
                    "confidence": min(0.88, max(0.58, float(fact.confidence or 0.0))),
                }
        return None

    def _find_condition_chunk_evidence(self, db: Session, plan_id: str, terms: tuple[str, ...]) -> dict | None:
        chunks = (
            db.execute(
                select(PolicyChunk)
                .where(PolicyChunk.plan_id == plan_id)
                .order_by(PolicyChunk.page_start.asc(), PolicyChunk.paragraph_index.asc())
            )
            .scalars()
            .all()
        )
        lower_terms = [t.lower() for t in terms if t]
        for c in chunks:
            text = c.text or ""
            if any(t in text.lower() for t in lower_terms):
                line = self._pick_line(text, lower_terms)
                return {
                    "value": line,
                    "page": c.page_start,
                    "section": c.section_path,
                    "quote": line[:300],
                    "confidence": 0.56,
                }
        return None

    def _find_condition_evidence_from_retrieval(
        self,
        evidence_rows: list[dict[str, Any]],
        terms: tuple[str, ...],
    ) -> dict | None:
        lower_terms = [t.lower() for t in terms if t]
        for row in evidence_rows:
            quote = str(row.get("quote") or "").strip()
            section = str(row.get("section") or "")
            merged = f"{quote} {section}".lower()
            if not quote:
                continue
            if lower_terms and not any(t in merged for t in lower_terms):
                continue
            return {
                "value": self._pick_line(quote, lower_terms),
                "page": row.get("page"),
                "section": row.get("section"),
                "quote": self._shorten(quote, 300),
                "confidence": 0.52,
            }
        return None

    def _pick_line(self, text: str, lower_terms: list[str]) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        for ln in lines:
            ll = ln.lower()
            if any(t in ll for t in lower_terms):
                return self._shorten(ln, 180)
        return self._shorten(lines[0] if lines else text, 180)

    def _compact_compare_for_llm(
        self,
        compare: CompareResponse,
        query: str,
        plan_name_map: dict[str, str],
    ) -> dict:
        c_key = condition_dimension_for_query(query)
        rows = []
        if c_key:
            rows.extend([r for r in compare.rows if r.dimension_key == c_key])
        rows.extend([r for r in compare.rows if r.is_different and (not c_key or r.dimension_key != c_key)])
        rows = rows[:6]

        compact_rows: list[dict] = []
        for r in rows:
            pv: dict[str, dict] = {}
            for pid, cell in r.plan_values.items():
                plan_name = plan_name_map.get(pid, pid)
                pv[plan_name] = {
                    "value": self._shorten(cell.value or "", 140),
                    "confidence": float(cell.confidence or 0.0),
                    "source": {
                        "page": cell.source.page if cell.source else None,
                        "section": cell.source.section if cell.source else None,
                    },
                }
            compact_rows.append(
                {
                    "dimension_key": r.dimension_key,
                    "dimension_label": r.dimension_label,
                    "is_different": bool(r.is_different),
                    "plan_values": pv,
                }
            )

        return {
            "plans": [plan_name_map.get(pid, pid) for pid in compare.plan_ids],
            "dimensions": compare.dimensions,
            "rows": compact_rows,
        }

    def _looks_insufficient(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return True
        keys = (
            "\u8bc1\u636e\u4e0d\u8db3",
            "\u65e0\u6cd5\u660e\u786e",
            "\u4fe1\u606f\u4e0d\u8db3",
            "insufficient",
            "not enough evidence",
        )
        return any(k in t for k in keys)

    def _replace_plan_ids_with_names(self, text: str, plan_name_map: dict[str, str]) -> str:
        out = text or ""
        for pid, name in plan_name_map.items():
            if pid and name:
                out = out.replace(pid, name)
        return out

    def _build_reply(
        self,
        added_plans: list[str],
        added_dims: list[str],
        compare: CompareResponse | None,
        query: str,
        plan_name_map: dict[str, str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
        selected_plan_ids: list[str],
        auto_discovery_status: str = "",
        auto_discovery_message: str = "",
    ) -> str:
        lines: list[str] = []
        if auto_discovery_message:
            lines.append(auto_discovery_message)
            if auto_discovery_status == "none":
                return "\n".join(lines)
        if added_plans:
            lines.append(f"\u5df2\u52a0\u5165\u8ba1\u5212: {', '.join(dict.fromkeys(added_plans))}")
        if added_dims:
            lines.append(f"\u5df2\u52a0\u5165\u7ef4\u5ea6: {', '.join(dict.fromkeys(added_dims))}")

        if len(selected_plan_ids) < 2:
            single_plan_reply = self._build_single_plan_evidence_reply(
                selected_plan_ids=selected_plan_ids,
                plan_name_map=plan_name_map,
                evidence_by_plan=evidence_by_plan,
            )
            if single_plan_reply:
                lines.append(single_plan_reply)
                return "\n".join(lines)
            if auto_discovery_status != "single":
                lines.append("\u8bf7\u81f3\u5c11\u9009\u62e9\u4e24\u4e2a\u8ba1\u5212\u540e\u518d\u8fdb\u884c\u6bd4\u8f83\u3002")
            return "\n".join(lines)

        if compare and compare.rows:
            diff_count = sum(1 for r in compare.rows if r.is_different)
            lines.append(
                f"\u5f53\u524d\u6bd4\u8f83\u5305\u542b {len(compare.plan_ids)} \u4e2a\u8ba1\u5212\u3001"
                f"{len(compare.rows)} \u4e2a\u7ef4\u5ea6\uff0c\u5176\u4e2d {diff_count} \u4e2a\u7ef4\u5ea6\u5b58\u5728\u660e\u663e\u5dee\u5f02\u3002"
            )
            compare_summary = self._build_compare_row_summary(
                query=query,
                compare=compare,
                selected_plan_ids=selected_plan_ids,
                plan_name_map=plan_name_map,
                preferred_dimension_keys=added_dims,
            )
            if compare_summary:
                lines.append(compare_summary)
                return "\n".join(lines)

        conclusion = self._build_evidence_conclusion(
            query=query,
            selected_plan_ids=selected_plan_ids,
            plan_name_map=plan_name_map,
            evidence_by_plan=evidence_by_plan,
        )
        if conclusion:
            lines.append(conclusion)
        return "\n".join(lines)

    def _build_single_plan_evidence_reply(
        self,
        *,
        selected_plan_ids: list[str],
        plan_name_map: dict[str, str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
    ) -> str:
        if len(selected_plan_ids) != 1:
            return ""
        pid = selected_plan_ids[0]
        rows = self._dedupe_evidence_rows(evidence_by_plan.get(pid, []), max_items=2)
        if not rows:
            return ""
        name = plan_name_map.get(pid, pid)
        out = [f"\u7ed3\u8bba: \u5f53\u524d\u68c0\u7d22\u5230 1 \u4e2a\u76f8\u5173\u8ba1\u5212 **{name}**\u3002", "\u4e3b\u8981\u4f9d\u636e:"]
        for row in rows:
            quote = self._highlight_key_numbers(self._shorten(str(row.get("quote") or ""), 140))
            src = self._format_evidence_source(row)
            out.append(f"- **{name}**: {quote}{src}")
        out.append(
            "\u51b3\u7b56\u63d0\u793a: \u5f53\u524d\u53ea\u6709 1 \u4e2a\u5339\u914d\u8ba1\u5212\uff0c"
            "\u82e5\u9700\u6a2a\u5411\u5bf9\u6bd4\uff0c\u53ef\u7ee7\u7eed\u68c0\u7d22\u5176\u4ed6\u8ba1\u5212\u3002"
        )
        return "\n".join(out)

    def _build_evidence_conclusion(
        self,
        query: str,
        selected_plan_ids: list[str],
        plan_name_map: dict[str, str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
    ) -> str:
        del query
        if len(selected_plan_ids) < 2:
            return ""

        scores: dict[str, float] = {}
        for pid in selected_plan_ids:
            evidences = evidence_by_plan.get(pid, [])
            scores[pid] = sum(float(e.get("score") or 0.0) for e in evidences)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return "\u7ed3\u8bba: \u672a\u68c0\u7d22\u5230\u53ef\u7528\u8bc1\u636e\uff0c\u8bf7\u5c1d\u8bd5\u66f4\u5177\u4f53\u7684\u95ee\u6cd5\u6216\u68c0\u67e5\u539f\u59cb\u4fdd\u5355\u6587\u6863\u3002"

        top_pid, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        gap = top_score - second_score
        gap_ratio = gap / max(1e-6, top_score)
        evidence_close = top_score <= 0.0 or gap_ratio < 0.18

        out: list[str] = []
        if evidence_close:
            out.append(
                f"\u7ed3\u8bba: \u5f53\u524d\u5bf9\u6bd4\u7684 {len(selected_plan_ids)} \u4e2a\u8ba1\u5212\u5728\u8be5\u95ee\u9898\u4e0a\u7684\u8bc1\u636e\u63a5\u8fd1\uff0c\u6682\u65e0\u6cd5\u7ed9\u51fa\u5355\u4e00\u6700\u4f18\u7ed3\u8bba\u3002"
            )
        else:
            top_name = plan_name_map.get(top_pid, top_pid)
            out.append(
                f"\u7ed3\u8bba: \u76ee\u524d\u8bc1\u636e\u66f4\u504f\u5411 **{top_name}**\uff0c"
                "\u4f46\u6700\u7ec8\u9009\u62e9\u4ecd\u5efa\u8bae\u7531\u4f60\u7ed3\u5408\u6761\u6b3e\u5b8c\u6574\u6027\u6765\u51b3\u5b9a\u3002"
            )

        out.append("\u4e3b\u8981\u4f9d\u636e:")
        for pid in selected_plan_ids[:4]:
            name = plan_name_map.get(pid, pid)
            evs = self._dedupe_evidence_rows(
                evidence_by_plan.get(pid, []),
                max_items=1,
            )
            if not evs:
                out.append(f"- **{name}**: \u672a\u68c0\u7d22\u5230\u76f4\u63a5\u8bc1\u636e")
                continue
            e = evs[0]
            src = self._format_evidence_source(e)
            quote = self._highlight_key_numbers(self._shorten(str(e.get("quote") or ""), 120))
            out.append(f"- **{name}**: {quote}{src}")

        out.append(
            "\u51b3\u7b56\u63d0\u793a: \u8bf7\u4f60\u6839\u636e\u4fdd\u8d39\u9884\u7b97\u3001\u7b49\u5f85\u671f\u3001\u9664\u5916\u6761\u6b3e\u3001"
            "\u7406\u8d54\u504f\u597d\u7b49\u7ef4\u5ea6\u81ea\u884c\u786e\u5b9a\u6700\u7ec8\u8ba1\u5212\u3002"
        )
        return "\n".join(out)

    def _build_compare_row_summary(
        self,
        query: str,
        compare: CompareResponse,
        selected_plan_ids: list[str],
        plan_name_map: dict[str, str],
        preferred_dimension_keys: list[str] | None = None,
    ) -> str:
        if not compare.rows:
            return ""

        target_keys = [k for k in (preferred_dimension_keys or []) if k in (compare.dimensions or [])]
        if not target_keys:
            target_keys = self._target_dimension_keys_for_query(query, compare)
        if not target_keys:
            return ""

        target_rows = [r for r in compare.rows if r.dimension_key in target_keys]
        if not target_rows:
            return ""

        parts: list[str] = []
        for row in target_rows[:2]:
            non_empty_plan_ids = [
                pid
                for pid in selected_plan_ids
                if self._has_meaningful_compare_value(row.plan_values.get(pid))
            ]
            if row.dimension_key.startswith("condition_"):
                conclusion = self._build_condition_row_conclusion(
                    row=row,
                    selected_plan_ids=selected_plan_ids,
                    plan_name_map=plan_name_map,
                    non_empty_plan_ids=non_empty_plan_ids,
                )
            else:
                if not non_empty_plan_ids:
                    parts.append(f"\u7ed3\u8bba: \u672a\u627e\u5230 {row.dimension_label} \u7684\u76f4\u63a5\u5bf9\u6bd4\u4fe1\u606f")
                    continue
                conclusion = f"\u7ed3\u8bba: \u57fa\u4e8e\u300c{row.dimension_label}\u300d\u5bf9\u6bd4\uff0c\u5404\u8ba1\u5212\u8981\u70b9\u5982\u4e0b\u3002"

            row_lines = [conclusion, "\u4e3b\u8981\u4f9d\u636e:"]
            for pid in selected_plan_ids[:4]:
                name = plan_name_map.get(pid, pid)
                cell = row.plan_values.get(pid)
                if not cell:
                    row_lines.append(f"- **{name}**: \u672a\u63d0\u53d6\u5230")
                    continue
                value = self._shorten(cell.value or "\u672a\u63d0\u53d6\u5230", 140)
                src = self._format_compare_cell_source(cell)
                row_lines.append(f"- **{name}**: {value}{src}")
            row_lines.append("\u51b3\u7b56\u63d0\u793a: \u8bf7\u4f18\u5148\u4ee5\u5bf9\u6bd4\u8868\u5bf9\u5e94\u7ef4\u5ea6\u4e3a\u51c6\uff0c\u518d\u7ed3\u5408\u6761\u6b3e\u7ec6\u5219\u786e\u8ba4\u3002")
            parts.append("\n".join(row_lines))
        return "\n".join(parts)

    def _has_meaningful_compare_value(self, cell: CellValue | None) -> bool:
        if cell is None:
            return False
        value = (cell.value or "").strip()
        if not value:
            return False
        missing_tokens = {
            "\u672a\u63d0\u53d6\u5230",
            "\u672a\u68c0\u7d22\u5230\u76f4\u63a5\u8bc1\u636e",
            "n/a",
            "na",
        }
        return value.lower() not in {t.lower() for t in missing_tokens}

    def _build_condition_row_conclusion(
        self,
        *,
        row: CompareRow,
        selected_plan_ids: list[str],
        plan_name_map: dict[str, str],
        non_empty_plan_ids: list[str],
    ) -> str:
        if not non_empty_plan_ids:
            return (
                f"\u7ed3\u8bba: \u5f53\u524d\u5df2\u9009\u8ba1\u5212\u4e2d\uff0c"
                f"\u5747\u672a\u63d0\u53d6\u5230\u300c{row.dimension_label}\u300d\u7684\u76f4\u63a5\u4fdd\u969c\u4fe1\u606f\u3002"
            )
        if len(non_empty_plan_ids) == 1:
            best_pid = non_empty_plan_ids[0]
            best_name = plan_name_map.get(best_pid, best_pid)
            return (
                f"\u7ed3\u8bba: \u5f53\u524d\u5df2\u9009\u8ba1\u5212\u4e2d\uff0c"
                f"**{best_name}** \u5bf9\u300c{row.dimension_label}\u300d\u6709\u660e\u786e\u4fdd\u969c\u63cf\u8ff0\uff0c\u53ef\u4f18\u5148\u8003\u8651\u8be5\u8ba1\u5212\u3002"
            )
        names = [plan_name_map.get(pid, pid) for pid in non_empty_plan_ids[:3]]
        return (
            f"\u7ed3\u8bba: \u5f53\u524d\u5df2\u9009\u8ba1\u5212\u4e2d\uff0c"
            f"{'\u3001'.join(names)} \u5747\u63d0\u5230\u300c{row.dimension_label}\u300d\u76f8\u5173\u4fdd\u969c\uff0c"
            "\u6682\u65e0\u5355\u4e00\u4f18\u9009\u8ba1\u5212\u3002"
        )

    def _target_dimension_keys_for_query(self, query: str, compare: CompareResponse) -> list[str]:
        detected = detect_dimensions(query)
        if detected:
            return [k for k in detected if k in (compare.dimensions or [])]
        c_key = condition_dimension_for_query(query)
        if c_key and c_key in (compare.dimensions or []):
            return [c_key]
        # Fallback: when compare currently focuses on one or two dimensions, use current table keys.
        if len(compare.dimensions or []) <= 2:
            return list(compare.dimensions or [])
        return []

    def _format_compare_cell_source(self, cell: CellValue) -> str:
        src = cell.source
        if not src:
            return ""
        parts: list[str] = []
        if src.page is not None:
            parts.append(f"p{src.page}")
        if src.section:
            parts.append(str(src.section))
        if not parts:
            return ""
        return f"\uff08\u6765\u6e90: {' / '.join(parts)}\uff09"

    def _filter_evidence_by_focus_terms(
        self,
        evidence_by_plan: dict[str, list[dict[str, Any]]],
        focus_terms: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        if not focus_terms:
            return evidence_by_plan
        out: dict[str, list[dict[str, Any]]] = {}
        lowered_terms = [t.lower() for t in focus_terms if t]
        for pid, rows in evidence_by_plan.items():
            matched: list[dict[str, Any]] = []
            for row in rows or []:
                quote = str(row.get("quote") or "").lower()
                section = str(row.get("section") or "").lower()
                merged = f"{quote} {section}"
                if any(t in merged for t in lowered_terms):
                    matched.append(row)
            out[pid] = matched
        return out

    def _format_evidence_source(self, evidence: dict[str, Any]) -> str:
        page = evidence.get("page")
        section = evidence.get("section")
        parts: list[str] = []
        if page is not None:
            parts.append(f"p{page}")
        if section:
            parts.append(str(section))
        if not parts:
            return ""
        return f"\uff08\u6765\u6e90: {' / '.join(parts)}\uff09"

    def _compact_evidence_for_llm(
        self,
        selected_plan_ids: list[str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
        plan_name_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for pid in selected_plan_ids:
            rows = self._dedupe_evidence_rows(evidence_by_plan.get(pid, []), max_items=1)
            compact_rows: list[dict[str, Any]] = []
            for row in rows:
                compact_rows.append(
                    {
                        "score": float(row.get("score") or 0.0),
                        "quote": self._shorten(str(row.get("quote") or ""), 180),
                        "page": row.get("page"),
                        "section": row.get("section"),
                        "from": row.get("from"),
                    }
                )
            payload.append(
                {
                    "plan_id": pid,
                    "plan_name": plan_name_map.get(pid, pid),
                    "evidence": compact_rows,
                }
            )
        return payload

    def _shorten(self, text: str, max_len: int) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 1] + "..."

    def _dedupe_evidence_rows(self, rows: list[dict[str, Any]], max_items: int = 2) -> list[dict[str, Any]]:
        if not rows:
            return []
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None, str | None]] = set()
        ranked = sorted(rows, key=lambda x: float(x.get("score") or 0.0), reverse=True)
        for row in ranked:
            quote = self._shorten(str(row.get("quote") or ""), 160)
            page = row.get("page")
            section = row.get("section")
            key = (quote, page, section)
            if key in seen:
                continue
            seen.add(key)
            row_copy = dict(row)
            row_copy["quote"] = quote
            out.append(row_copy)
            if len(out) >= max_items:
                break
        return out

    def _highlight_key_numbers(self, text: str) -> str:
        if not text:
            return text
        highlighted = re.sub(
            r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:\u7f8e\u5143|\u6e2f\u5143|USD|HKD|\u500d|%))",
            r"**\1**",
            text,
        )
        return highlighted

    def _state_out(self, state: SessionState) -> SessionStateOut:
        return SessionStateOut(
            session_id=state.session_id,
            selected_plans=state.selected_plans or [],
            dimensions=state.dimensions or [],
            filters=state.filters or {},
            updated_at=state.updated_at,
        )

