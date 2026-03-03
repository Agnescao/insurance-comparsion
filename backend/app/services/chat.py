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
from app.services.llm_planner import ConditionIntentResult, LLMPlanner, PlannerResult


@dataclass
class AutoDiscoveryOutcome:
    triggered: bool = False
    status: str = ""
    message: str = ""
    selected_plan_ids: list[str] = field(default_factory=list)


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
        auto_discovery = self._auto_discover_when_uncovered(db, state, plans, content, plan_name_map)
        enforce_min_plans = not (auto_discovery.triggered and auto_discovery.status in {"none", "single"})
        self._ensure_min_plans(state, plans, enforce=enforce_min_plans)
        compare = self._build_compare_if_possible(db, state, content)
        evidence_by_plan = self.retriever.retrieve_plan_evidence(
            db,
            query=content,
            plan_ids=state.selected_plans or [],
            per_plan_k=2,
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
        skip_llm_for_table_grounded_query = bool(detect_dimensions(content) or condition_dimension_for_query(content))
        if not auto_discovery.triggered and not skip_llm_for_table_grounded_query:
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

    def _build_compare_if_possible(self, db: Session, state: SessionState, content: str) -> CompareResponse | None:
        if len(state.selected_plans or []) < 2:
            return None
        if not (state.dimensions or []):
            return None
        compare = self.compare_service.build_compare(
            db,
            plan_ids=state.selected_plans,
            dimensions=state.dimensions,
            filters=state.filters,
        )
        evidence_by_plan: dict[str, list[dict[str, Any]]] = {}
        if condition_dimension_for_query(content):
            evidence_by_plan = self.retriever.retrieve_plan_evidence(
                db,
                query=content,
                plan_ids=state.selected_plans or [],
                per_plan_k=2,
            )
        self._backfill_condition_row_from_chunks(
            db,
            compare,
            state.selected_plans,
            content,
            evidence_by_plan=evidence_by_plan,
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
        if compare is None or len(state.selected_plans or []) < 2:
            return ""
        payload = self._compact_evidence_for_llm(state.selected_plans or [], evidence_by_plan, plan_name_map)
        if not payload:
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
            "推荐其他计划",
            "推荐新计划",
            "找新计划",
            "找更多计划",
            "加入计划",
            "添加计划",
            "扩展对比",
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

    def _auto_discover_when_uncovered(
        self,
        db: Session,
        state: SessionState,
        plans: list[Plan],
        query: str,
        plan_name_map: dict[str, str],
    ) -> AutoDiscoveryOutcome:
        selected = list(state.selected_plans or [])
        if len(selected) < 2:
            return AutoDiscoveryOutcome()

        llm_intent: ConditionIntentResult | None = self.planner.parse_condition_intent(
            query=query,
            state={
                "selected_plans": selected,
                "dimensions": state.dimensions or [],
                "filters": state.filters or {},
            },
        )
        is_condition_query = (
            llm_intent.is_condition_or_surgery if llm_intent is not None else self._is_condition_or_surgery_query(query)
        )
        if not is_condition_query:
            return AutoDiscoveryOutcome()

        focus_terms = list(llm_intent.focus_terms or []) if llm_intent is not None else []
        if not focus_terms:
            focus_terms = self._extract_focus_terms(query)
        if not focus_terms:
            return AutoDiscoveryOutcome()
        signal_terms = self._high_signal_focus_terms(focus_terms)
        target = self._target_phrase_for_message(query, signal_terms)
        semantic_query = " ".join(signal_terms[:3]) if signal_terms else query
        candidate_ids = [p.plan_id for p in plans if p.plan_id not in selected]
        if not candidate_ids:
            msg = f"目前未找到覆盖「{target}」的计划。"
            return AutoDiscoveryOutcome(triggered=True, status="none", message=msg)

        matched: list[str] = []
        # Preferred path: model-driven coverage judgement on semantic retrieval evidence.
        if llm_intent is not None:
            selected_evidence = self.retriever.retrieve_plan_evidence(
                db,
                query=semantic_query,
                plan_ids=selected,
                per_plan_k=3,
            )
            selected_plans_payload = [
                {"plan_id": pid, "name": plan_name_map.get(pid, pid)}
                for pid in selected
            ]
            selected_covered = self.planner.assess_query_coverage(
                query=query,
                plans=selected_plans_payload,
                evidence_by_plan=selected_evidence,
            )
            if selected_covered:
                return AutoDiscoveryOutcome()

            discovered = self.retriever.discover_plan_ids(
                db,
                query=semantic_query,
                top_k=max(6, len(candidate_ids)),
                candidate_plan_ids=candidate_ids,
            )
            discovered = [pid for pid in discovered if pid in candidate_ids]
            if discovered:
                discovered_payload = [{"plan_id": pid, "name": plan_name_map.get(pid, pid)} for pid in discovered]
                discovered_evidence = self.retriever.retrieve_plan_evidence(
                    db,
                    query=semantic_query,
                    plan_ids=discovered,
                    per_plan_k=3,
                )
                covered = self.planner.assess_query_coverage(
                    query=query,
                    plans=discovered_payload,
                    evidence_by_plan=discovered_evidence,
                )
                if covered is not None:
                    covered_set = set(covered)
                    matched = [pid for pid in discovered if pid in covered_set]

        # Fallback path: lexical validation when model path unavailable/unreliable.
        if not matched:
            corpus_cache: dict[str, list[str]] = {}
            selected_hit = any(self._plan_contains_focus_terms(db, pid, signal_terms, corpus_cache) for pid in selected)
            if selected_hit:
                return AutoDiscoveryOutcome()

            discovered = self.retriever.discover_plan_ids(
                db,
                query=semantic_query,
                top_k=max(6, len(candidate_ids)),
                candidate_plan_ids=candidate_ids,
            )
            discovered = [pid for pid in discovered if pid in candidate_ids]
            matched = [pid for pid in discovered if self._plan_contains_focus_terms(db, pid, signal_terms, corpus_cache)]
            if len(matched) < 2:
                fallback_matched = [
                    pid for pid in candidate_ids if self._plan_contains_focus_terms(db, pid, signal_terms, corpus_cache)
                ]
                if fallback_matched:
                    merged = matched + [pid for pid in fallback_matched if pid not in matched]
                    matched = merged

        if not matched:
            msg = f"目前未找到覆盖「{target}」的计划。"
            return AutoDiscoveryOutcome(triggered=True, status="none", message=msg)

        deduped = list(dict.fromkeys(matched))
        if len(deduped) == 1:
            pid = deduped[0]
            state.selected_plans = [pid]
            name = plan_name_map.get(pid, pid)
            msg = f"发现 1 个覆盖「{target}」的计划: {name}。当前仅 1 个计划，暂不生成对比表。"
            return AutoDiscoveryOutcome(
                triggered=True,
                status="single",
                message=msg,
                selected_plan_ids=[pid],
            )

        selected_new = deduped[:4]
        state.selected_plans = selected_new
        names = [plan_name_map.get(pid, pid) for pid in selected_new]
        msg = (
            f"在其他产品中发现 {len(selected_new)} 个覆盖「{target}」的计划，"
            f"已自动替换当前对比计划: {', '.join(names)}。"
        )
        return AutoDiscoveryOutcome(
            triggered=True,
            status="multi",
            message=msg,
            selected_plan_ids=selected_new,
        )

    def _high_signal_focus_terms(self, focus_terms: list[str]) -> list[str]:
        if not focus_terms:
            return []
        anchors = ("手术", "手術", "切除", "移植", "介入", "ectomy", "otomy", "癌", "瘤", "病", "症", "炎")
        signal = [t for t in focus_terms if any(a.lower() in t.lower() for a in anchors)]
        if signal:
            return list(dict.fromkeys(signal))
        return focus_terms

    def _is_condition_or_surgery_query(self, query: str) -> bool:
        c_key = condition_dimension_for_query(query)
        if c_key:
            return True
        text = (query or "").strip()
        lowered = text.lower()
        keywords = (
            "手术",
            "手術",
            "切除",
            "移植",
            "搭桥",
            "搭橋",
            "介入",
            "surgery",
            "operation",
            "ectomy",
            "otomy",
            "癌",
            "肿瘤",
            "腫瘤",
            "瘤",
            "疾病",
            "病症",
            "cancer",
            "disease",
            "tumor",
        )
        if any(k.lower() in lowered for k in keywords):
            return True

        # Fallback: explicit coverage intent + disease/surgery-like entity mention.
        intent_tokens = ("保障", "覆盖", "cover", "理赔", "赔付", "友好", "适合", "更好")
        if any(tok in lowered for tok in intent_tokens):
            if re.search(
                r"[A-Za-z0-9\u4e00-\u9fff]{2,24}(?:癌|病|症|瘤|炎|手术|手術|切除|移植|介入|ectomy|otomy)",
                text,
                flags=re.IGNORECASE,
            ):
                return True
            if self._extract_focus_terms(query):
                return True
        return False

    def _extract_focus_terms(self, query: str) -> list[str]:
        text = (query or "").strip()
        lowered = text.lower()
        terms: list[str] = []

        for cond_terms in CONDITION_TERMS.values():
            for term in cond_terms:
                if term and term.lower() in lowered:
                    terms.append(term.lower())

        intent_entity_matches = re.findall(
            r"(?:对|對|关于|關於|针对|針對)\s*([A-Za-z0-9\u4e00-\u9fff]{2,24}?(?:手术|手術|切除|移植|介入|癌|病|症|瘤|炎))",
            text,
            flags=re.IGNORECASE,
        )
        terms.extend([m.lower() for m in intent_entity_matches if m])

        surgery_matches = re.findall(
            r"([A-Za-z0-9\u4e00-\u9fff]{2,16}(?:切除|移植|搭桥|搭橋|介入|成形术|成形術|置换|置換)(?:手术|手術)?)",
            text,
            flags=re.IGNORECASE,
        )
        terms.extend([m.lower() for m in surgery_matches if m])
        surgery_label_matches = re.findall(
            r"([A-Za-z0-9\u4e00-\u9fff]{2,14}(?:手术|手術))",
            text,
            flags=re.IGNORECASE,
        )
        terms.extend([m.lower() for m in surgery_label_matches if m])
        english_surgery_matches = re.findall(
            r"([a-z0-9][a-z0-9\-\s]{1,24}(?:ectomy|otomy|surgery|operation))",
            lowered,
            flags=re.IGNORECASE,
        )
        terms.extend([m.lower() for m in english_surgery_matches if m])

        disease_matches = re.findall(
            r"([A-Za-z0-9\u4e00-\u9fff]{2,16}(?:癌|病|症|瘤))",
            text,
            flags=re.IGNORECASE,
        )
        terms.extend([m.lower() for m in disease_matches if m])

        if not terms:
            for generic in ("手术", "手術", "surgery", "operation", "癌症", "癌", "疾病", "病症"):
                if generic.lower() in lowered:
                    terms.append(generic.lower())

        expanded: list[str] = []
        for term in terms:
            cleaned = self._clean_focus_term(term)
            if cleaned:
                expanded.extend(self._expand_focus_terms(cleaned))
        deduped = list(dict.fromkeys([t for t in expanded if len(t) >= 2]))
        deduped = self._best_focus_terms(deduped)
        return deduped[:12]

    def _clean_focus_term(self, term: str) -> str:
        t = (term or "").strip().lower()
        if not t:
            return ""
        t = re.sub(r"[，。；;,.!?？！\s]+", "", t)
        t = re.sub(r"^(那有什么计划|那有什麼計劃|有什么计划|有什麼計劃|有啥计划|哪些计划|哪种计划|哪種計劃)+", "", t)
        t = re.sub(r"^(如果|如|请问|请帮我|请帮忙|帮我|我要|我想|想要|做|进行|接受)+", "", t)
        t = re.sub(r"^(哪个计划|哪款计划|哪一个计划|哪个|哪款|哪一个)+", "", t)
        t = re.sub(r"^(在我患|患有|患了|关于|針對|针对|对于|對|对)+", "", t)
        marker_match = re.search(r"(?:對|对|關於|关于|針對|针对)([A-Za-z0-9\u4e00-\u9fff]{2,24})", t)
        if marker_match:
            t = marker_match.group(1)
        t = re.sub(r"(哪个计划|哪款计划|哪一个计划|更友好|更好|比较好|怎么样|如何|是否|吗)$", "", t)
        t = re.sub(r"(有保障|有保險|有覆盖|有覆蓋|可保障|可覆蓋|保障如何|怎么保障|怎麼保障|保障)$", "", t)
        t = t.strip()
        if len(t) < 2:
            return ""
        return t

    def _best_focus_terms(self, terms: list[str]) -> list[str]:
        if not terms:
            return []
        generic = {"手术", "手術", "surgery", "operation", "疾病", "病症", "癌", "癌症", "cancer", "disease"}
        scored: list[tuple[int, str]] = []
        for term in terms:
            score = 0
            if term not in generic:
                score += 4
            if 2 <= len(term) <= 12:
                score += 3
            if any(k in term for k in ("切除", "移植", "搭桥", "搭橋", "介入", "癌", "肿瘤", "腫瘤")):
                score += 3
            if any(k in term for k in ("计划", "保单", "保障", "如果", "哪个")):
                score -= 3
            scored.append((score, term))
        scored.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
        ordered = [t for _, t in scored]
        return list(dict.fromkeys(ordered))

    def _expand_focus_terms(self, term: str) -> list[str]:
        out = [term]
        for suffix in ("手术", "手術"):
            if term.endswith(suffix):
                base = term[: -len(suffix)].strip()
                if len(base) >= 2:
                    out.append(base)
        for suffix in ("surgery", "operation"):
            if term.endswith(suffix):
                base = term[: -len(suffix)].strip()
                if len(base) >= 2:
                    out.append(base)
        replacements = (
            ("手术", "手術"),
            ("手術", "手术"),
            ("结肠", "結腸"),
            ("結腸", "结肠"),
            ("心脏", "心臟"),
            ("心臟", "心脏"),
            ("肿瘤", "腫瘤"),
            ("腫瘤", "肿瘤"),
            ("卵巢恶性肿瘤", "卵巢惡性腫瘤"),
            ("卵巢惡性腫瘤", "卵巢恶性肿瘤"),
        )
        for src, dst in replacements:
            if src in term:
                out.append(term.replace(src, dst))
        char_replacements = (
            ("结", "結"),
            ("肠", "腸"),
            ("状", "狀"),
            ("疗", "療"),
            ("术", "術"),
            ("门", "門"),
            ("诊", "診"),
            ("风", "風"),
            ("脑", "腦"),
            ("肾", "腎"),
            ("关", "關"),
            ("节", "節"),
            ("颈", "頸"),
            ("动", "動"),
            ("脉", "脈"),
            ("尔", "爾"),
            ("茨", "茲"),
            ("呆", "癡"),
            ("瘫", "癱"),
        )
        for src, dst in char_replacements:
            if src in term:
                out.append(term.replace(src, dst))
            if dst in term:
                out.append(term.replace(dst, src))
        return out

    def _plan_contains_focus_terms(
        self,
        db: Session,
        plan_id: str,
        focus_terms: list[str],
        corpus_cache: dict[str, list[str]],
    ) -> bool:
        if not focus_terms:
            return False
        if plan_id not in corpus_cache:
            corpus_cache[plan_id] = self._load_plan_corpus(db, plan_id)
        corpus = corpus_cache[plan_id]
        normalized_terms = list(
            dict.fromkeys(
                [
                    t.strip().lower()
                    for term in focus_terms
                    for t in (term, self._normalize_medical_text(term))
                    if t and len(t.strip()) >= 2
                ]
            )
        )
        return any(self._term_matches_text(term, text) for term in normalized_terms for text in corpus)

    def _term_matches_text(self, term: str, text: str) -> bool:
        t = (term or "").strip().lower()
        x = (text or "").strip().lower()
        if not t or not x:
            return False
        if t in x:
            return True
        if t.endswith("手术") or t.endswith("手術"):
            base = t[:-2]
            if len(base) >= 2 and re.search(
                re.escape(base) + r"[A-Za-z0-9\u4e00-\u9fff]{0,6}(?:手术|手術)",
                x,
                flags=re.IGNORECASE,
            ):
                return True
        if t.endswith("surgery"):
            base = t[: -len("surgery")].strip()
            if len(base) >= 2 and re.search(
                re.escape(base) + r"[a-z0-9\-\s]{0,12}(?:surgery|operation)",
                x,
                flags=re.IGNORECASE,
            ):
                return True
        return False

    def _load_plan_corpus(self, db: Session, plan_id: str) -> list[str]:
        lines: list[str] = []
        facts = (
            db.execute(select(PolicyFact).where(PolicyFact.plan_id == plan_id))
            .scalars()
            .all()
        )
        for fact in facts:
            lines.extend(
                [
                    str(fact.dimension_key or "").lower(),
                    str(fact.dimension_label or "").lower(),
                    str(fact.condition_text or "").lower(),
                    str(fact.value_text or "").lower(),
                    str(fact.normalized_value or "").lower(),
                    str(fact.source_quote or "").lower(),
                ]
            )

        chunks = (
            db.execute(select(PolicyChunk).where(PolicyChunk.plan_id == plan_id))
            .scalars()
            .all()
        )
        for chunk in chunks:
            if chunk.text:
                lines.append(str(chunk.text).lower())
        out: list[str] = []
        for ln in lines:
            if not ln:
                continue
            out.append(ln)
            normalized = self._normalize_medical_text(ln)
            if normalized and normalized != ln:
                out.append(normalized)
        return out

    def _normalize_medical_text(self, text: str) -> str:
        raw = (text or "").strip().lower()
        if not raw:
            return ""
        canon_map = {
            "結": "结",
            "腸": "肠",
            "狀": "状",
            "療": "疗",
            "術": "术",
            "門": "门",
            "診": "诊",
            "風": "风",
            "腦": "脑",
            "腎": "肾",
            "關": "关",
            "節": "节",
            "頸": "颈",
            "動": "动",
            "脈": "脉",
            "腫": "肿",
            "惡": "恶",
            "爾": "尔",
            "茲": "茨",
            "癡": "呆",
            "癱": "瘫",
        }
        chars = [canon_map.get(ch, ch) for ch in raw]
        joined = "".join(chars)
        return re.sub(r"[\W_]+", "", joined, flags=re.UNICODE)

    def _target_phrase_for_message(self, query: str, focus_terms: list[str]) -> str:
        if focus_terms:
            generic = {"手术", "手術", "surgery", "operation", "疾病", "病症", "癌", "癌症", "cancer", "disease"}
            preferred = [
                t.strip()
                for t in focus_terms
                if t
                and t.strip()
                and t.strip() not in generic
                and not any(x in t for x in ("计划", "有什么", "什麼", "那有", "哪个", "哪個"))
            ]
            if preferred:
                preferred.sort(key=len)
                return preferred[0]
            term = focus_terms[0].strip()
            if term:
                return term
        return self._shorten((query or "").strip(), 24) or "该疾病/手术"

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
            if auto_discovery_status == "single":
                return "\n".join(lines)
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

    def _build_evidence_conclusion(
        self,
        query: str,
        selected_plan_ids: list[str],
        plan_name_map: dict[str, str],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
    ) -> str:
        if len(selected_plan_ids) < 2:
            return ""

        focus_terms = self._extract_focus_terms(query)
        filtered_evidence_by_plan = self._filter_evidence_by_focus_terms(evidence_by_plan, focus_terms)
        scores: dict[str, float] = {}
        for pid in selected_plan_ids:
            evidences = filtered_evidence_by_plan.get(pid, []) or evidence_by_plan.get(pid, [])
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
                filtered_evidence_by_plan.get(pid, []) or evidence_by_plan.get(pid, []),
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
