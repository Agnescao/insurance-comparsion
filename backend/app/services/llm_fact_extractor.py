from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import settings
from app.services.chunking import ChunkDoc
from app.services.dimensions import CONDITION_TERMS, DIMENSIONS, dimension_label
from app.services.fact_extractor import FactExtractor, FactRecord

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


class LLMFactExtractor:
    KEY_COMPARE_DIMENSIONS: tuple[str, ...] = (
        "coverage_hospitalization",
        "coverage_outpatient",
        "coverage_surgery",
        "premium_payment",
        "annual_limit",
        "itemized_limit",
        "deductible_copay",
        "riders_benefits",
        "exclusions",
    )
    OUTPATIENT_REQUIRED_TERMS: tuple[str, ...] = (
        "门诊",
        "門診",
        "诊症",
        "診症",
        "outpatient",
        "clinic",
        "day case",
    )
    OUTPATIENT_NOISE_TERMS: tuple[str, ...] = (
        "疾病",
        "病症",
        "保障疾病",
        "癌症",
        "肿瘤",
        "心脏病",
        "中风",
        "赔偿一览",
    )

    def __init__(self, *, mode: str = "llm", fallback: FactExtractor | None = None) -> None:
        self.logger = logging.getLogger("uvicorn.error")
        self.mode = (mode or "llm").strip().lower()
        self.fallback = fallback or FactExtractor()

        key = (
            settings.fact_extractor_api_key
            or settings.dashscope_api_key
            or settings.llm_api_key
            or settings.qwen_api_key
            or ""
        )
        self.api_key = key.strip()
        self.model = settings.fact_extractor_model
        self.base_url = settings.fact_extractor_base_url
        self.timeout_sec = int(settings.fact_extractor_timeout_sec)
        self.max_tokens = int(settings.fact_extractor_max_tokens)

        self.enabled = bool(self.mode in {"llm", "hybrid"} and self.api_key and OpenAI is not None)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.enabled else None
        self._runtime_unavailable = False

        if self.mode in {"llm", "hybrid"} and not self.enabled:
            reason = "missing_api_key" if not self.api_key else "openai_not_installed"
            self.logger.warning("fact.llm.disabled mode=%s reason=%s", self.mode, reason)

    def extract_from_chunks(
        self,
        chunks: list[ChunkDoc],
        *,
        plan_name: str | None = None,
        source_file: str | None = None,
    ) -> list[FactRecord]:
        if not chunks:
            return []

        if not self.enabled:
            return self.fallback.extract_from_chunks(chunks, plan_name=plan_name, source_file=source_file)
        if self._runtime_unavailable:
            return self.fallback.extract_from_chunks(chunks, plan_name=plan_name, source_file=source_file)

        all_facts: list[FactRecord] = []
        for idx, chunk in enumerate(chunks):
            text = (chunk.text or "").strip()
            if not text:
                continue
            try:
                all_facts.extend(
                    self._extract_chunk(
                        chunk=chunk,
                        chunk_index=idx,
                        plan_name=plan_name,
                        source_file=source_file,
                    )
                )
            except Exception as exc:
                self.logger.warning(
                    "fact.llm.chunk.failed model=%s page=%s chunk_index=%s err=%s",
                    self.model,
                    chunk.page_start,
                    idx,
                    exc,
                )
                self._runtime_unavailable = True
                self.logger.warning("fact.llm.circuit_open model=%s fallback=rule", self.model)
                break

        if not all_facts:
            self.logger.warning("fact.llm.empty_facts fallback_to_rule plan=%s", plan_name or "")
            return self.fallback.extract_from_chunks(chunks, plan_name=plan_name, source_file=source_file)

        merged_facts = self._maybe_backfill_with_rules(
            all_facts=all_facts,
            chunks=chunks,
            plan_name=plan_name,
            source_file=source_file,
        )
        return self.fallback.post_process_facts(merged_facts)

    def _extract_chunk(
        self,
        *,
        chunk: ChunkDoc,
        chunk_index: int,
        plan_name: str | None,
        source_file: str | None,
    ) -> list[FactRecord]:
        assert self.client is not None

        messages = self._build_messages(
            chunk=chunk,
            chunk_index=chunk_index,
            plan_name=plan_name,
            source_file=source_file,
        )
        self.logger.info(
            "fact.llm.call.start model=%s plan=%s page=%s chunk_index=%s text_chars=%d",
            self.model,
            plan_name or "",
            chunk.page_start,
            chunk_index,
            len(chunk.text or ""),
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=self.max_tokens,
            timeout=self.timeout_sec,
        )
        content = (completion.choices[0].message.content or "").strip()
        usage = getattr(completion, "usage", None)
        self.logger.info(
            "fact.llm.call.done model=%s chunk_index=%s response_chars=%d prompt_tokens=%s completion_tokens=%s",
            self.model,
            chunk_index,
            len(content),
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

        parsed = self._parse_json(content)
        raw_facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
        if not isinstance(raw_facts, list):
            raw_facts = []
        allowed_keys = {d.key for d in DIMENSIONS} | set(CONDITION_TERMS.keys())

        out: list[FactRecord] = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue

            dim_key = str(item.get("dimension_key") or "").strip()
            if not dim_key:
                continue
            if dim_key not in allowed_keys:
                continue
            value_text = str(item.get("value_text") or "").strip()
            if not value_text:
                continue
            dim_label = str(item.get("dimension_label") or dimension_label(dim_key)).strip()
            normalized_value = item.get("normalized_value")
            if normalized_value is not None:
                normalized_value = str(normalized_value)[:2048]
            numeric_value = self._coerce_float(item.get("numeric_value"))
            unit = item.get("unit")
            unit = str(unit).strip() if unit is not None else None
            condition_text = item.get("condition_text")
            condition_text = str(condition_text).strip() if condition_text is not None else None
            source_quote = str(item.get("source_quote") or value_text).strip()[:300]
            plausibility_text = " ".join([value_text, condition_text or "", source_quote])
            if not self._is_dimension_fact_plausible(dim_key, plausibility_text):
                continue
            confidence = self._coerce_confidence(item.get("confidence"), default=0.72)

            metadata = item.get("metadata_json")
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": str(metadata)}
            metadata["extractor"] = "llm_qwen"
            metadata["chunk_index"] = chunk_index
            if dim_key == "premium_payment":
                premiums = self._extract_money_options(value_text)
                if premiums:
                    metadata["annual_premium_values"] = premiums
                modes = self._extract_payment_modes(value_text)
                if modes:
                    metadata["payment_modes"] = modes
                years = self._extract_year_options(value_text)
                if years:
                    metadata["payment_terms_years"] = years
            elif dim_key in {"annual_limit", "itemized_limit"}:
                limits = self._extract_money_options(value_text)
                if limits:
                    metadata["limit_values"] = limits

            out.append(
                FactRecord(
                    dimension_key=dim_key,
                    dimension_label=dim_label,
                    value_text=value_text,
                    normalized_value=normalized_value,
                    numeric_value=numeric_value,
                    unit=unit,
                    condition_text=condition_text,
                    confidence=confidence,
                    source_page=chunk.page_start,
                    source_section=chunk.section_path,
                    source_quote=source_quote,
                    metadata_json=metadata,
                )
            )

        self.logger.info(
            "fact.llm.chunk.parsed model=%s page=%s chunk_index=%s fact_count=%d",
            self.model,
            chunk.page_start,
            chunk_index,
            len(out),
        )
        return out

    def _is_dimension_fact_plausible(self, dimension_key: str, value_text: str) -> bool:
        text = (value_text or "").strip().lower()
        if not text:
            return False
        if dimension_key == "coverage_outpatient":
            has_required = any(term.lower() in text for term in self.OUTPATIENT_REQUIRED_TERMS)
            has_noise = any(term.lower() in text for term in self.OUTPATIENT_NOISE_TERMS)
            return has_required and not (has_noise and not has_required)
        return True

    def _build_messages(
        self,
        *,
        chunk: ChunkDoc,
        chunk_index: int,
        plan_name: str | None,
        source_file: str | None,
    ) -> list[dict[str, str]]:
        dimensions = [{"key": d.key, "label": d.label} for d in DIMENSIONS]
        conditions = [
            {
                "dimension_key": condition_key,
                "label": dimension_label(condition_key),
                "terms": list(terms),
            }
            for condition_key, terms in CONDITION_TERMS.items()
        ]
        schema = {
            "facts": [
                {
                    "dimension_key": "string",
                    "dimension_label": "string",
                    "value_text": "string",
                    "normalized_value": "string|null",
                    "numeric_value": "number|null",
                    "unit": "string|null",
                    "condition_text": "string|null",
                    "confidence": "number(0-1)",
                    "source_quote": "string",
                    "metadata_json": {"any": "object"},
                }
            ]
        }
        dimension_hints = [
            {"dimension_key": "coverage_hospitalization", "hints": ["住院", "住院病房", "住院费用", "inpatient"]},
            {"dimension_key": "coverage_outpatient", "hints": ["门诊", "門診", "outpatient", "clinic"]},
            {"dimension_key": "coverage_surgery", "hints": ["手术", "手術", "operation", "surgery"]},
            {
                "dimension_key": "premium_payment",
                "hints": [
                    "保费",
                    "保費",
                    "缴费",
                    "繳費",
                    "年度保费",
                    "年度保費",
                    "annual premium",
                    "每年保费",
                    "每年保費",
                    "年缴/半年缴/季缴/月缴",
                    "年繳/半年繳/季繳/月繳",
                    "保费缴付期",
                ],
            },
            {
                "dimension_key": "annual_limit",
                "hints": [
                    "年度限额",
                    "每年限额",
                    "annual limit",
                    "个人最高赔偿限额",
                    "個人最高賠償限額",
                    "最高赔偿限额",
                ],
            },
            {
                "dimension_key": "itemized_limit",
                "hints": [
                    "分项限额",
                    "分項限額",
                    "子限额",
                    "sub-limit",
                    "每项赔偿上限",
                    "每項賠償上限",
                    "个人最高赔偿限额",
                    "個人最高賠償限額",
                ],
            },
            {"dimension_key": "deductible_copay", "hints": ["自付额", "自負額", "共付", "共負", "deductible", "copay"]},
            {"dimension_key": "riders_benefits", "hints": ["附加险", "附加福利", "附加契约", "rider", "benefit"]},
            {"dimension_key": "exclusions", "hints": ["除外责任", "除外責任", "免责", "免責", "exclusion", "waiting period"]},
        ]
        system_prompt = (
            "You are an insurance policy fact extraction engine. "
            "Return JSON only, with no markdown and no extra text. "
            "Extract structured facts for comparison table dimensions. "
            "Rules: "
            "1) Use only facts explicitly present in chunk text. "
            "2) Use only provided dimension_key values. "
            "3) Prioritize key dimensions: coverage_hospitalization, coverage_outpatient, coverage_surgery, "
            "premium_payment, annual_limit, itemized_limit, deductible_copay, riders_benefits, exclusions. "
            "4) For premium_payment, prioritize annual premium amount and payment mode, NOT only payment term years. "
            "If annual premium exists, value_text must include both annual premium and payment mode. "
            "Good example: 年度保费：2,180美元；缴费方式：年缴/半年缴；缴费期：10/18/25年. "
            "Bad example: 3种保费缴付期. "
            "5) For annual_limit and itemized_limit, aggressively capture personal max compensation caps. "
            "Always capture concrete numbers if present, especially phrases like "
            "个人最高赔偿限额为400,000港元 / 50,000美元. "
            "6) Prefer one concise, information-dense fact per dimension per chunk. "
            "Avoid generic long marketing paragraphs if a numeric clause exists nearby. "
            "6.1) coverage_outpatient MUST explicitly mention outpatient/clinic/门诊 context. "
            "Do not output disease lists as coverage_outpatient. "
            "6) source_quote must come from original chunk text. "
            "7) confidence in [0,1]. "
            "8) If no facts, return {\"facts\":[]}."
        )
        user_payload = {
            "task": "extract_insurance_facts_by_dimensions",
            "plan_name": plan_name,
            "source_file": source_file,
            "chunk_index": chunk_index,
            "page_start": chunk.page_start,
            "section_path": chunk.section_path,
            "dimensions": dimensions,
            "dimension_hints": dimension_hints,
            "condition_dimensions": conditions,
            "response_schema": schema,
            "chunk_text": chunk.text,
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def _parse_json(self, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {"facts": []}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {"facts": []}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"facts": []}

    def _coerce_confidence(self, value: Any, *, default: float) -> float:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return default
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _coerce_float(self, value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_year_options(self, text: str) -> list[int]:
        values: set[int] = set()
        for token in re.findall(r"(?<!\d)(\d{1,2})\s*[年\u5e74]", text or ""):
            year = int(token)
            if 5 <= year <= 40:
                values.add(year)
        return sorted(values)

    def _extract_money_options(self, text: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(
            r"(?:HK\$|US\$|\$|USD|HKD|RMB|CNY|\u6e2f\u5143|\u7f8e\u5143|\u4eba\u6c11\u5e01)?\s*"
            r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*"
            r"(HKD|USD|RMB|CNY|\u6e2f\u5143|\u7f8e\u5143|\u4eba\u6c11\u5e01)?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text or ""):
            raw_num = (m.group(1) or "").replace(",", "").strip()
            if not raw_num:
                continue
            try:
                amount = float(raw_num)
            except ValueError:
                continue
            if amount < 300:
                continue
            unit = (m.group(2) or "").strip()
            val = f"{int(amount):,}" if amount.is_integer() else str(amount)
            val = f"{val}{unit}" if unit else val
            if val in seen:
                continue
            seen.add(val)
            values.append(val)
            if len(values) >= 3:
                break
        return values

    def _extract_payment_modes(self, text: str) -> list[str]:
        patterns: tuple[tuple[str, str], ...] = (
            ("年缴", r"年缴|年繳|年付|annual"),
            ("半年缴", r"半年缴|半年繳|半年付|semi-annual|semi annual"),
            ("季缴", r"季缴|季繳|季付|quarterly"),
            ("月缴", r"月缴|月繳|月付|monthly"),
            ("趸缴", r"趸缴|躉繳|single premium"),
        )
        found: list[str] = []
        lowered = (text or "").lower()
        for mode, pattern in patterns:
            if re.search(pattern, lowered, flags=re.IGNORECASE):
                found.append(mode)
        return found

    def _maybe_backfill_with_rules(
        self,
        *,
        all_facts: list[FactRecord],
        chunks: list[ChunkDoc],
        plan_name: str | None,
        source_file: str | None,
    ) -> list[FactRecord]:
        llm_dim_set = {f.dimension_key for f in all_facts}
        missing_dims = [k for k in self.KEY_COMPARE_DIMENSIONS if k not in llm_dim_set]

        premium_facts = [f for f in all_facts if f.dimension_key == "premium_payment"]
        has_annual_premium = any((f.metadata_json or {}).get("annual_premium_values") for f in premium_facts)
        has_payment_modes = any((f.metadata_json or {}).get("payment_modes") for f in premium_facts)
        if premium_facts and (not has_annual_premium or not has_payment_modes):
            if "premium_payment" not in missing_dims:
                missing_dims.append("premium_payment")

        low_coverage = len(llm_dim_set.intersection(set(self.KEY_COMPARE_DIMENSIONS))) < 4
        if not missing_dims and not low_coverage:
            return all_facts

        self.logger.warning(
            "fact.llm.coverage.low plan=%s llm_dim_count=%d missing=%s premium_has_amount=%s premium_has_mode=%s",
            plan_name or "",
            len(llm_dim_set),
            missing_dims,
            has_annual_premium,
            has_payment_modes,
        )

        rule_facts = self.fallback.extract_from_chunks(chunks, plan_name=plan_name, source_file=source_file)
        if not rule_facts:
            return all_facts

        merged = list(all_facts)
        added = 0
        include_dims = set(missing_dims)
        if low_coverage:
            include_dims.update(self.KEY_COMPARE_DIMENSIONS)
        for fact in rule_facts:
            if fact.dimension_key not in include_dims:
                continue
            metadata = dict(fact.metadata_json or {})
            metadata["extractor"] = "rule_backfill"
            metadata["backfill_reason"] = "llm_missing_dimension"
            merged.append(
                FactRecord(
                    dimension_key=fact.dimension_key,
                    dimension_label=fact.dimension_label,
                    value_text=fact.value_text,
                    normalized_value=fact.normalized_value,
                    numeric_value=fact.numeric_value,
                    unit=fact.unit,
                    condition_text=fact.condition_text,
                    confidence=max(float(fact.confidence), 0.74),
                    source_page=fact.source_page,
                    source_section=fact.source_section,
                    source_quote=fact.source_quote,
                    metadata_json=metadata,
                )
            )
            added += 1

        self.logger.info(
            "fact.llm.backfill.done plan=%s llm_facts=%d rule_facts=%d merged=%d added=%d",
            plan_name or "",
            len(all_facts),
            len(rule_facts),
            len(merged),
            added,
        )
        return merged
