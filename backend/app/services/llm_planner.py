from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import requests

from app.config import settings
from app.services.dimensions import detect_dimensions

logger = logging.getLogger("uvicorn.error")


@dataclass
class PlannerResult:
    mode: str
    actions: list[dict[str, Any]]
    reasoning: str | None = None


@dataclass
class ConditionIntentResult:
    is_condition_or_surgery: bool
    focus_terms: list[str]
    confidence: float = 0.0


class LLMPlanner:
    def __init__(self) -> None:
        self.api_key = (settings.llm_api_key or settings.qwen_api_key or "").strip()
        self.enabled = bool(self.api_key and (settings.llm_enabled or settings.llm_api_key or settings.qwen_api_key))

    def plan(
        self,
        query: str,
        state: dict[str, Any],
        available_plans: list[dict[str, str]],
    ) -> PlannerResult:
        if not self.enabled:
            return self._fallback_plan(query)

        prompt = self._planner_prompt(query=query, state=state, available_plans=available_plans)
        try:
            content = self._chat_completion(
                model=settings.llm_planner_model,
                messages=prompt,
                temperature=0.1,
                json_mode=True,
                max_tokens=320,
            )
            parsed = self._parse_json(content)
            if not parsed:
                return self._fallback_plan(query)

            mode = str(parsed.get("mode", "context_compare"))
            actions = parsed.get("actions", [])
            if not isinstance(actions, list):
                actions = []
            clean_actions = [a for a in actions if isinstance(a, dict) and a.get("type")]
            if not clean_actions:
                return self._fallback_plan(query)
            return PlannerResult(mode=mode, actions=clean_actions, reasoning=parsed.get("reasoning"))
        except Exception as exc:
            logger.warning("llm.plan.failed fallback_to_rule err=%s", exc)
            return self._fallback_plan(query)

    def summarize_compare(
        self,
        query: str,
        compare_payload: dict[str, Any] | None,
        state_payload: dict[str, Any],
    ) -> str:
        if not self.enabled:
            return ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an insurance comparison assistant. "
                    "Use only the provided compare JSON. "
                    "Return concise Chinese text with three parts: "
                    "1) 结论 2) 主要依据(2-3条) 3) 决策提示. "
                    "Do not fabricate facts."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "state": state_payload,
                        "compare": compare_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            text = self._chat_completion(
                model=settings.llm_answer_model,
                messages=messages,
                temperature=0.1,
                json_mode=False,
                max_tokens=420,
            )
            return text.strip()
        except Exception as exc:
            logger.warning("llm.summarize_compare.failed err=%s", exc)
            return ""

    def summarize_evidence_compare(
        self,
        query: str,
        plan_names: list[str],
        evidence_payload: list[dict[str, Any]],
        state_payload: dict[str, Any],
    ) -> str:
        if not self.enabled:
            return ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an insurance comparison assistant. "
                    "Use ONLY the provided evidence JSON. No hallucination. "
                    "Return strict JSON only with this schema: "
                    "{"
                    "\"evidence_close\": true/false,"
                    "\"recommended_plan\": \"string or empty\","
                    "\"conclusion\": \"short Chinese sentence\","
                    "\"evidence\": ["
                    "  {\"plan\": \"plan name\", \"summary\": \"one concise sentence\", \"page\": 1}"
                    "],"
                    "\"decision_tip\": \"short Chinese sentence\""
                    "}. "
                    "Rules: "
                    "1) one evidence item per plan, no duplicates; "
                    "2) include page if available; "
                    "3) summary should include key numbers when present; "
                    "4) never include file paths."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "plans": plan_names,
                        "state": state_payload,
                        "evidence_by_plan": evidence_payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            text = self._chat_completion(
                model=settings.llm_answer_model,
                messages=messages,
                temperature=0.1,
                json_mode=True,
                max_tokens=520,
            )
            parsed = self._parse_json(text)
            if not parsed:
                return ""
            return self._render_evidence_summary(parsed, plan_names)
        except Exception as exc:
            logger.warning("llm.summarize_evidence.failed err=%s", exc)
            return ""

    def compare_dimension_from_evidence(
        self,
        *,
        query: str,
        dimension_key: str,
        dimension_label: str,
        plans: list[dict[str, str]],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        evidence_counts = {pid: len(rows or []) for pid, rows in evidence_by_plan.items()}
        logger.info(
            "llm.compare_dimension.start dim=%s plans=%d evidence_counts=%s",
            dimension_key,
            len(plans),
            evidence_counts,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You compare insurance plans by one dimension. "
                    "Use only evidence JSON and return strict JSON only: "
                    "{"
                    "\"dimension_key\":\"string\","
                    "\"is_different\":true/false,"
                    "\"plan_values\":{"
                    "  \"<plan_id>\":{\"value\":\"summary\", \"confidence\":0.0, \"evidence_index\":0}"
                    "}"
                    "}. "
                    "If evidence is weak, set value to 证据不足."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "dimension_key": dimension_key,
                        "dimension_label": dimension_label,
                        "plans": plans,
                        "evidence_by_plan": evidence_by_plan,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            content = self._chat_completion(
                model=settings.llm_answer_model,
                messages=messages,
                temperature=0.1,
                json_mode=True,
                max_tokens=500,
            )
            parsed = self._parse_json(content)
            if not parsed or not isinstance(parsed, dict):
                logger.warning("llm.compare_dimension.invalid_json dim=%s", dimension_key)
                return None

            plan_values = parsed.get("plan_values", {})
            if not isinstance(plan_values, dict):
                logger.warning("llm.compare_dimension.invalid_plan_values dim=%s", dimension_key)
                return None

            cleaned_values: dict[str, dict[str, Any]] = {}
            for p in plans:
                pid = str(p.get("plan_id") or "")
                if not pid:
                    continue
                raw = plan_values.get(pid, {})
                if not isinstance(raw, dict):
                    raw = {}
                value = str(raw.get("value") or "证据不足").strip() or "证据不足"
                confidence = self._coerce_confidence(raw.get("confidence"))
                evidence_index = self._coerce_int(raw.get("evidence_index"))
                cleaned_values[pid] = {
                    "value": value,
                    "confidence": confidence,
                    "evidence_index": evidence_index,
                }

            out = {
                "dimension_key": str(parsed.get("dimension_key") or dimension_key),
                "is_different": bool(parsed.get("is_different")),
                "plan_values": cleaned_values,
            }
            logger.info("llm.compare_dimension.done dim=%s is_different=%s", dimension_key, out["is_different"])
            return out
        except Exception as exc:
            logger.warning("llm.compare_dimension.failed dim=%s err=%s", dimension_key, exc)
            return None

    def parse_condition_intent(self, *, query: str, state: dict[str, Any] | None = None) -> ConditionIntentResult | None:
        if not self.enabled:
            return None
        state = state or {}
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intent parser for insurance compare chat. "
                    "Return strict JSON only with schema: "
                    "{"
                    "\"is_condition_or_surgery\": true/false,"
                    "\"focus_terms\": [\"term1\", \"term2\"],"
                    "\"confidence\": 0.0"
                    "}. "
                    "Rules: "
                    "1) focus_terms must be concrete disease/surgery names only, no generic words. "
                    "2) remove wording like '哪个计划/对我/更友好/有保障'. "
                    "3) if no concrete target, return empty focus_terms and is_condition_or_surgery=false."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "state": state,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            text = self._chat_completion(
                model=settings.llm_planner_model,
                messages=messages,
                temperature=0.0,
                json_mode=True,
                max_tokens=220,
            )
            parsed = self._parse_json(text)
            if not parsed or not isinstance(parsed, dict):
                return None
            is_flag = bool(parsed.get("is_condition_or_surgery"))
            raw_terms = parsed.get("focus_terms", [])
            if not isinstance(raw_terms, list):
                raw_terms = []
            clean_terms: list[str] = []
            for x in raw_terms:
                term = str(x or "").strip()
                if not term:
                    continue
                if len(term) > 32:
                    term = term[:32]
                if term not in clean_terms:
                    clean_terms.append(term)
            confidence = self._coerce_confidence(parsed.get("confidence"))
            return ConditionIntentResult(
                is_condition_or_surgery=is_flag,
                focus_terms=clean_terms[:6],
                confidence=confidence,
            )
        except Exception as exc:
            logger.warning("llm.intent_parse.failed fallback_to_rule err=%s", exc)
            return None

    def assess_query_coverage(
        self,
        *,
        query: str,
        plans: list[dict[str, str]],
        evidence_by_plan: dict[str, list[dict[str, Any]]],
    ) -> list[str] | None:
        if not self.enabled:
            return None
        if not plans:
            return []

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evidence judge for insurance coverage questions. "
                    "Use ONLY the provided evidence text. Return strict JSON only: "
                    "{"
                    "\"covered_plan_ids\": [\"plan_id\"],"
                    "\"confidence\": 0.0"
                    "}. "
                    "Rules: "
                    "1) Mark covered only if evidence explicitly indicates relevant disease/surgery coverage. "
                    "2) Generic marketing text is NOT coverage evidence. "
                    "3) If uncertain, do not mark covered."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "plans": plans,
                        "evidence_by_plan": evidence_by_plan,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            text = self._chat_completion(
                model=settings.llm_answer_model,
                messages=messages,
                temperature=0.0,
                json_mode=True,
                max_tokens=260,
            )
            parsed = self._parse_json(text)
            if not parsed or not isinstance(parsed, dict):
                return None
            raw = parsed.get("covered_plan_ids", [])
            if not isinstance(raw, list):
                raw = []
            allowed = {str(p.get("plan_id") or "") for p in plans if p.get("plan_id")}
            covered: list[str] = []
            for pid in raw:
                key = str(pid or "").strip()
                if key and key in allowed and key not in covered:
                    covered.append(key)
            return covered
        except Exception as exc:
            logger.warning("llm.coverage_assess.failed fallback_to_rule err=%s", exc)
            return None

    def _planner_prompt(
        self,
        query: str,
        state: dict[str, Any],
        available_plans: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        schema = {
            "mode": "context_compare|product_discovery",
            "actions": [
                {"type": "add_dimensions", "dimensions": ["dimension_key"]},
                {"type": "set_filters", "filters": {}},
                {"type": "set_compare_plans", "plan_ids": ["plan_id"]},
                {"type": "discover_products", "query": "string", "top_k": 3},
                {"type": "refresh_compare"},
            ],
            "reasoning": "optional short note",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a planning agent for insurance compare chat. "
                    "Return strict JSON only. "
                    "Rules: "
                    "1) Default mode is context_compare. "
                    "2) Use product_discovery and discover_products ONLY when user explicitly asks for NEW plans "
                    "(e.g., 推荐其他计划, 找新计划, more plans, add another plan). "
                    "3) If user asks follow-up about currently selected plans, NEVER discover new plans. "
                    "4) If disease/condition mentioned, include add_dimensions action. "
                    "5) Always include refresh_compare as last action."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "state": state,
                        "available_plans": available_plans,
                        "response_schema": schema,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

    def _render_evidence_summary(self, parsed: dict[str, Any], plan_names: list[str]) -> str:
        evidence_close = bool(parsed.get("evidence_close"))
        recommended_plan = str(parsed.get("recommended_plan") or "").strip()
        conclusion = str(parsed.get("conclusion") or "").strip()
        decision_tip = str(parsed.get("decision_tip") or "").strip()

        evidence_rows = parsed.get("evidence", [])
        if not isinstance(evidence_rows, list):
            evidence_rows = []

        by_plan: dict[str, dict[str, Any]] = {}
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            plan = str(row.get("plan") or "").strip()
            if not plan or plan in by_plan:
                continue
            summary = str(row.get("summary") or "").strip()
            page = self._coerce_int(row.get("page"))
            by_plan[plan] = {"summary": summary, "page": page}

        lines: list[str] = []
        if not conclusion:
            if evidence_close:
                conclusion = "两个计划在该问题上的证据接近，暂无法给出单一最优结论。"
            elif recommended_plan:
                conclusion = f"目前证据更偏向 **{recommended_plan}**，但建议结合完整条款决策。"
            else:
                conclusion = "目前证据不足以给出单一最优计划。"
        else:
            if recommended_plan and recommended_plan in conclusion:
                conclusion = conclusion.replace(recommended_plan, f"**{recommended_plan}**")
        lines.append(f"结论: {conclusion}")

        lines.append("主要依据:")
        for name in plan_names:
            row = by_plan.get(name)
            if not row:
                lines.append(f"- **{name}**: 未检索到直接证据")
                continue
            summary = self._highlight_numbers(row.get("summary") or "")
            page = row.get("page")
            if page is not None:
                lines.append(f"- **{name}**: {summary}（页码{page}）")
            else:
                lines.append(f"- **{name}**: {summary}")

        if not decision_tip:
            decision_tip = "请结合预算、等待期、除外条款和理赔偏好做最终选择。"
        lines.append(f"决策提示: {decision_tip}")
        return "\n".join(lines)

    def _highlight_numbers(self, text: str) -> str:
        if not text:
            return text
        return re.sub(
            r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?\s*(?:美元|港元|USD|HKD|倍|%))",
            r"**\1**",
            text,
        )

    def _chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        logger.info(
            "llm.chat_completion.start model=%s json_mode=%s messages=%d max_tokens=%s",
            model,
            json_mode,
            len(messages),
            max_tokens,
        )
        response = requests.post(url, headers=headers, json=payload, timeout=settings.llm_timeout_sec)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        logger.info("llm.chat_completion.done model=%s response_chars=%d", model, len(content or ""))
        return content

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        raw = text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _fallback_plan(self, query: str) -> PlannerResult:
        dims = detect_dimensions(query)
        actions: list[dict[str, Any]] = []
        if dims:
            actions.append({"type": "add_dimensions", "dimensions": dims})

        lowered = (query or "").lower()
        discovery_keywords = (
            "推荐其他",
            "推荐新计划",
            "找新计划",
            "更多计划",
            "add another plan",
            "recommend other plans",
            "find new plan",
        )
        if any(k in lowered for k in discovery_keywords):
            actions.append({"type": "discover_products", "query": query, "top_k": 3})
            mode = "product_discovery"
        else:
            mode = "context_compare"

        actions.append({"type": "refresh_compare"})
        return PlannerResult(mode=mode, actions=actions)

    def _coerce_confidence(self, value: Any) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f

    def _coerce_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None
