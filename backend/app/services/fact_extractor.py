from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.chunking import ChunkDoc
from app.services.dimensions import CONDITION_TERMS, DIMENSIONS, dimension_label


@dataclass
class FactRecord:
    dimension_key: str
    dimension_label: str
    value_text: str
    normalized_value: str | None
    numeric_value: float | None
    unit: str | None
    condition_text: str | None
    confidence: float
    source_page: int | None
    source_section: str | None
    source_quote: str | None
    metadata_json: dict[str, object] | None = None


MONEY_PATTERN = re.compile(
    r"(?:HK\$|US\$|\$|USD|HKD|RMB|CNY)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(HKD|USD|RMB|CNY|%|times|days|years)?",
    re.IGNORECASE,
)
PAYMENT_TERM_KEYWORDS = (
    "\u4fdd\u8d39\u7f34\u4ed8\u671f",
    "\u4fdd\u8cbb\u7e73\u4ed8\u671f",
    "\u4fdd\u8d39\u7f34\u8d39\u671f",
    "\u4fdd\u8cbb\u7e73\u8cbb\u671f",
    "\u7f34\u4ed8\u671f",
    "\u7e73\u4ed8\u671f",
    "\u7f34\u8d39\u671f",
    "\u7e73\u8cbb\u671f",
    "\u7f34\u8d39\u5e74\u671f",
    "\u7e73\u8cbb\u5e74\u671f",
    "\u7f34\u4ed8\u5e74\u671f",
    "\u7e73\u4ed8\u5e74\u671f",
)
PREMIUM_HINT_KEYWORDS = (
    "\u4fdd\u8d39",
    "\u4fdd\u8cbb",
    "\u7f34\u8d39",
    "\u7e73\u8cbb",
    "\u7f34\u4ed8",
    "\u7e73\u4ed8",
    "payment",
    "premium",
)
PAYMENT_MODE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("\u5e74\u7f34", re.compile(r"\u5e74\u7f34|\u5e74\u7e73|\u5e74\u4ed8|annual", re.IGNORECASE)),
    (
        "\u534a\u5e74\u7f34",
        re.compile(r"\u534a\u5e74\u7f34|\u534a\u5e74\u7e73|\u534a\u5e74\u4ed8|semi-annual|semi annual", re.IGNORECASE),
    ),
    ("\u5b63\u7f34", re.compile(r"\u5b63\u7f34|\u5b63\u7e73|\u5b63\u4ed8|quarterly", re.IGNORECASE)),
    ("\u6708\u7f34", re.compile(r"\u6708\u7f34|\u6708\u7e73|\u6708\u4ed8|monthly", re.IGNORECASE)),
    ("\u8e89\u7f34", re.compile(r"\u8e89\u7f34|\u8e89\u7e73|\u8e89\u4ed8|single premium", re.IGNORECASE)),
)
GENERIC_NOISE_TERMS = (
    "\u8be6\u60c5\u8bf7\u53c2\u9605",
    "\u8bf7\u53c2\u9605",
    "\u4ec5\u4f9b\u53c2\u8003",
    "\u53ea\u4f9b\u53c2\u8003",
    "\u514d\u8d23\u58f0\u660e",
    "\u6211\u4eec\u4fdd\u7559",
    "\u4fdd\u8d39\u8c03\u6574",
    "\u4fdd\u8cbb\u8abf\u6574",
    "\u76d1\u7ba1\u5c40",
    "\u793a\u4f8b",
    "\u5047\u8bbe",
    "\u6ce8\u610f",
)
PAYMENT_LIST_PATTERN = re.compile(r"((?:\d{1,2}\s*[/\uFF0F\u3001,\uFF0C]\s*)+\d{1,2})\s*\u5e74")
YEAR_VALUE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})\s*\u5e74")
ANNUAL_PREMIUM_LABEL_PATTERN = re.compile(
    r"(?:\u5e74\u5ea6\u4fdd\u8d39|\u5e74\u7f34\u4fdd\u8d39|annual premium|\u5e74\u5ea6\u4fdd\u8cbb|\u5e74\u7e73\u4fdd\u8cbb)",
    re.IGNORECASE,
)
MONEY_VALUE_PATTERN = re.compile(
    r"(?:HK\$|US\$|\$|USD|HKD|RMB|CNY|\u6e2f\u5143|\u7f8e\u5143|\u4eba\u6c11\u5e01)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(HKD|USD|RMB|CNY|\u6e2f\u5143|\u7f8e\u5143|\u4eba\u6c11\u5e01)?",
    re.IGNORECASE,
)
LIMIT_HINT_PATTERN = re.compile(
    r"\u4e2a\u4eba\u6700\u9ad8\u8d54\u507f\u9650\u989d|\u500b\u4eba\u6700\u9ad8\u8ce0\u511f\u9650\u984d|"
    r"\u6700\u9ad8\u8d54\u507f\u9650\u989d|\u6700\u9ad8\u8ce0\u511f\u9650\u984d|"
    r"\u5e74\u5ea6\u9650\u989d|\u6bcf\u5e74\u9650\u989d|annual limit|"
    r"\u5206\u9879\u9650\u989d|\u5206\u9805\u9650\u984d|\u6bcf\u9879\u8d54\u507f\u4e0a\u9650|\u6bcf\u9805\u8ce0\u511f\u4e0a\u9650",
    re.IGNORECASE,
)
OUTPATIENT_REQUIRED_TERMS = (
    "门诊",
    "門診",
    "诊症",
    "診症",
    "outpatient",
    "clinic",
    "day case",
)
OUTPATIENT_NOISE_TERMS = (
    "疾病",
    "病症",
    "保障疾病",
    "癌症",
    "肿瘤",
    "心脏病",
    "中风",
    "赔偿一览",
)


class FactExtractor:
    def __init__(self) -> None:
        self.dimension_patterns: dict[str, list[re.Pattern[str]]] = {}
        for d in DIMENSIONS:
            pats = [re.compile(re.escape(k), re.IGNORECASE) for k in d.keywords if k]
            self.dimension_patterns[d.key] = pats

    def extract_from_chunks(
        self,
        chunks: list[ChunkDoc],
        *,
        plan_name: str | None = None,
        source_file: str | None = None,
    ) -> list[FactRecord]:
        del plan_name, source_file  # reserved for extractor interface compatibility
        facts: list[FactRecord] = []
        for chunk in chunks:
            lines = [line.strip() for line in (chunk.text or "").splitlines() if line.strip()]
            if not lines:
                continue

            for dim in DIMENSIONS:
                if dim.key == "premium_payment":
                    premium_fact = self._extract_premium_fact(lines, chunk)
                    if premium_fact:
                        facts.append(premium_fact)
                        continue
                if dim.key in {"annual_limit", "itemized_limit"}:
                    limit_fact = self._extract_limit_fact(lines, chunk, dim.key)
                    if limit_fact:
                        facts.append(limit_fact)
                        continue

                sample = self._select_best_dimension_line(lines, dim.key)
                if not sample:
                    continue
                num, unit = self._extract_numeric(sample)
                facts.append(
                    FactRecord(
                        dimension_key=dim.key,
                        dimension_label=dim.label,
                        value_text=sample,
                        normalized_value=sample.lower(),
                        numeric_value=num,
                        unit=unit,
                        condition_text=None,
                        confidence=0.72 if num is not None else 0.62,
                        source_page=chunk.page_start,
                        source_section=chunk.section_path,
                        source_quote=sample[:300],
                    )
                )

            for condition_key, terms in CONDITION_TERMS.items():
                disease_lines = [line for line in lines if any(t.lower() in line.lower() for t in terms)]
                if not disease_lines:
                    continue
                sample = disease_lines[0]
                num, unit = self._extract_numeric(sample)
                facts.append(
                    FactRecord(
                        dimension_key=condition_key,
                        dimension_label=dimension_label(condition_key),
                        value_text=sample,
                        normalized_value=sample.lower(),
                        numeric_value=num,
                        unit=unit,
                        condition_text=terms[0],
                        confidence=0.66,
                        source_page=chunk.page_start,
                        source_section=chunk.section_path,
                        source_quote=sample[:300],
                    )
                )

        return self.post_process_facts(facts)

    def post_process_facts(self, facts: list[FactRecord]) -> list[FactRecord]:
        facts = self._filter_implausible_dimension_facts(facts)
        facts = self._append_plan_level_premium_summary(facts)
        facts = self._append_dimension_list_aggregates(facts)
        return self._dedup(facts)

    def _filter_implausible_dimension_facts(self, facts: list[FactRecord]) -> list[FactRecord]:
        out: list[FactRecord] = []
        for fact in facts:
            plausibility_text = " ".join(
                [
                    str(fact.value_text or ""),
                    str(fact.condition_text or ""),
                    str(fact.source_quote or ""),
                ]
            )
            if not self._is_dimension_fact_plausible(fact.dimension_key, plausibility_text):
                continue
            out.append(fact)
        return out

    def _is_dimension_fact_plausible(self, dimension_key: str, value_text: str) -> bool:
        text = (value_text or "").strip().lower()
        if not text:
            return False
        if dimension_key == "coverage_outpatient":
            has_required = any(term.lower() in text for term in OUTPATIENT_REQUIRED_TERMS)
            has_noise = any(term.lower() in text for term in OUTPATIENT_NOISE_TERMS)
            # Avoid mapping disease-list sentences into outpatient coverage.
            return has_required and not (has_noise and not has_required)
        return True

    def _line_match(self, line: str, dimension_key: str) -> bool:
        patterns = self.dimension_patterns.get(dimension_key, [])
        return any(pattern.search(line) for pattern in patterns)

    def _select_best_dimension_line(self, lines: list[str], dimension_key: str) -> str | None:
        candidates: list[tuple[int, str]] = [
            (idx, line) for idx, line in enumerate(lines) if self._line_match(line, dimension_key)
        ]
        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda x: (self._score_candidate_line(x[1], dimension_key), -x[0]),
            reverse=True,
        )
        return ranked[0][1]

    def _score_candidate_line(self, line: str, dimension_key: str) -> int:
        score = 0
        length = len(line)
        if 8 <= length <= 80:
            score += 3
        elif length > 140:
            score -= 4
        elif length < 4:
            score -= 2

        if re.search(r"\d", line):
            score += 2
        if any(x in line for x in (":", "\uff1a", "\u5305\u62ec", "\u63d0\u4f9b", "\u53ef\u9009", "\u9009\u62e9")):
            score += 2
        if any(x in line for x in GENERIC_NOISE_TERMS):
            score -= 4

        if dimension_key == "premium_payment":
            if self._contains_payment_term_hint(line):
                score += 8
            if any(x in line for x in ("\u5e74\u7f34", "\u534a\u5e74\u7f34", "\u5b63\u7f34", "\u6708\u7f34", "\u8e89\u7f34")):
                score += 5
            if "\u8c41\u514d" in line:
                score -= 1
        if dimension_key in {"annual_limit", "itemized_limit"}:
            if LIMIT_HINT_PATTERN.search(line):
                score += 8
            if any(x in line for x in ("HK$", "US$", "\u6e2f\u5143", "\u7f8e\u5143", "USD", "HKD")):
                score += 5

        return score

    def _extract_premium_fact(self, lines: list[str], chunk: ChunkDoc) -> FactRecord | None:
        annual_premiums, annual_line = self._extract_annual_premium_values(lines)
        years, evidence_line = self._extract_payment_terms(lines)
        modes = self._extract_payment_modes(lines)

        if not annual_premiums and not years and not modes:
            return None

        value_parts: list[str] = []
        normalized_parts: list[str] = []
        if annual_premiums:
            value_parts.append("\u5e74\u5ea6\u4fdd\u8d39\uff1a" + " / ".join(annual_premiums))
            normalized_parts.append("annual_premium_values:" + ",".join(annual_premiums))
        if modes:
            value_parts.append("\u7f34\u8d39\u65b9\u5f0f\uff1a" + " / ".join(modes))
            normalized_parts.append("payment_modes:" + ",".join(modes))
        if years and not annual_premiums:
            value_parts.append("\u4fdd\u8d39\u7f34\u4ed8\u671f\uff1a" + " / ".join(f"{y}\u5e74" for y in years))
            normalized_parts.append("payment_terms_years:" + ",".join(str(y) for y in years))

        summary_text = "\uff1b".join(value_parts)
        confidence = 0.9
        if annual_premiums:
            confidence = 0.96
        elif len(years) >= 2:
            confidence = 0.98
        elif years:
            confidence = 0.94
        elif modes:
            confidence = 0.82

        quote = annual_line or evidence_line or self._select_best_dimension_line(lines, "premium_payment") or lines[0]
        metadata: dict[str, object] = {}
        if annual_premiums:
            metadata["annual_premium_values"] = annual_premiums
        if years:
            metadata["payment_terms_years"] = years
        if modes:
            metadata["payment_modes"] = modes

        numeric_value = self._extract_numeric_from_money_text(annual_premiums[0]) if annual_premiums else None
        return FactRecord(
            dimension_key="premium_payment",
            dimension_label=dimension_label("premium_payment"),
            value_text=summary_text,
            normalized_value="|".join(normalized_parts) if normalized_parts else summary_text.lower(),
            numeric_value=numeric_value if numeric_value is not None else (float(years[0]) if years else None),
            unit=self._extract_unit_from_money_text(annual_premiums[0]) if annual_premiums else ("years" if years else None),
            condition_text=None,
            confidence=confidence,
            source_page=chunk.page_start,
            source_section=chunk.section_path,
            source_quote=quote[:300],
            metadata_json=metadata or None,
        )

    def _extract_annual_premium_values(self, lines: list[str]) -> tuple[list[str], str | None]:
        values: list[str] = []
        seen: set[str] = set()
        evidence: str | None = None
        for idx, line in enumerate(lines):
            if not ANNUAL_PREMIUM_LABEL_PATTERN.search(line):
                continue
            window = lines[idx : min(len(lines), idx + 4)]
            for probe in window:
                for m in MONEY_VALUE_PATTERN.finditer(probe):
                    amount = (m.group(1) or "").replace(",", "").strip()
                    if not amount:
                        continue
                    try:
                        amount_num = float(amount)
                    except ValueError:
                        continue
                    unit = (m.group(2) or "").strip()
                    raw = m.group(0) or ""
                    has_currency = bool(unit) or any(x in raw for x in ("HK$", "US$", "$", "\u6e2f\u5143", "\u7f8e\u5143", "\u4eba\u6c11\u5e01"))
                    if not has_currency and amount_num < 300:
                        continue
                    if "\u5e74\u4fdd\u8d39\u7f34\u4ed8\u671f" in probe or "\u5e74\u4fdd\u8cbb\u7e73\u4ed8\u671f" in probe:
                        continue

                    formatted_amount = f"{int(amount_num):,}" if float(amount_num).is_integer() else str(amount_num)
                    money_text = f"{formatted_amount}{unit}" if unit else formatted_amount
                    if money_text in seen:
                        continue
                    seen.add(money_text)
                    values.append(money_text)
                    evidence = evidence or probe
                    if len(values) >= 3:
                        return values, evidence
        return values, evidence

    def _extract_limit_fact(self, lines: list[str], chunk: ChunkDoc, dim_key: str) -> FactRecord | None:
        candidates: list[tuple[str, str]] = []
        for idx, line in enumerate(lines):
            if not LIMIT_HINT_PATTERN.search(line):
                continue
            window = lines[idx : min(len(lines), idx + 2)]
            merged = " ".join(window)
            candidates.append((line, merged))

        if not candidates:
            sample = self._select_best_dimension_line(lines, dim_key)
            if not sample:
                return None
            candidates.append((sample, sample))

        best_line = ""
        best_merged = ""
        best_caps: list[str] = []
        for line, merged in candidates:
            caps = self._extract_limit_caps(merged)
            if len(caps) > len(best_caps):
                best_line = line
                best_merged = merged
                best_caps = caps

        if not best_line:
            best_line, best_merged = candidates[0]

        label = "年度限额" if dim_key == "annual_limit" else "分项限额"
        if best_caps:
            value_text = f"{label}：" + " / ".join(best_caps[:4])
            confidence = 0.95
            numeric_value = self._extract_numeric_from_money_text(best_caps[0])
            unit = self._extract_unit_from_money_text(best_caps[0])
            metadata = {"limit_values": best_caps[:4]}
        else:
            value_text = best_line
            confidence = 0.72
            numeric_value, unit = self._extract_numeric(best_line)
            metadata = {"limit_values": []}

        return FactRecord(
            dimension_key=dim_key,
            dimension_label=dimension_label(dim_key),
            value_text=value_text,
            normalized_value=value_text.lower(),
            numeric_value=numeric_value,
            unit=unit,
            condition_text=None,
            confidence=confidence,
            source_page=chunk.page_start,
            source_section=chunk.section_path,
            source_quote=best_line[:300],
            metadata_json=metadata,
        )

    def _extract_limit_caps(self, text: str) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for m in MONEY_VALUE_PATTERN.finditer(text or ""):
            amount_raw = (m.group(1) or "").replace(",", "").strip()
            if not amount_raw:
                continue
            try:
                amount = float(amount_raw)
            except ValueError:
                continue
            if amount < 1000:
                continue

            unit = (m.group(2) or "").strip()
            raw_match = m.group(0) or ""
            if not unit:
                if "HK$" in raw_match:
                    unit = "HKD"
                elif "US$" in raw_match:
                    unit = "USD"
                elif "港元" in raw_match:
                    unit = "港元"
                elif "美元" in raw_match:
                    unit = "美元"

            amount_text = f"{int(amount):,}" if amount.is_integer() else str(amount)
            token = f"{amount_text}{unit}" if unit else amount_text
            if token in seen:
                continue
            seen.add(token)
            values.append(token)
            if len(values) >= 4:
                break
        return values

    def _extract_payment_terms(self, lines: list[str]) -> tuple[list[int], str | None]:
        years: set[int] = set()
        evidence_line: str | None = None

        for idx, line in enumerate(lines):
            if not self._contains_payment_term_hint(line):
                continue
            window_end = min(len(lines), idx + 7)
            for probe in lines[idx:window_end]:
                found = self._extract_year_values(probe)
                if found:
                    years.update(found)
                    evidence_line = evidence_line or line

        for line in lines:
            if not self._contains_premium_hint(line):
                continue
            if "\u5e74" not in line:
                continue
            found = self._extract_year_values(line)
            if found:
                years.update(found)
                evidence_line = evidence_line or line

        return sorted(years), evidence_line

    def _extract_year_values(self, text: str) -> list[int]:
        found: set[int] = set()

        for grouped in PAYMENT_LIST_PATTERN.findall(text):
            for token in re.split(r"[/\uFF0F\u3001,\uFF0C]", grouped):
                value = token.strip()
                if not value.isdigit():
                    continue
                year = int(value)
                if 5 <= year <= 40:
                    found.add(year)

        for value in YEAR_VALUE_PATTERN.findall(text):
            year = int(value)
            if 5 <= year <= 40:
                found.add(year)

        if not found and self._contains_payment_term_hint(text):
            for raw in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", text):
                year = int(raw)
                if 5 <= year <= 40:
                    found.add(year)

        return sorted(found)

    def _extract_payment_modes(self, lines: list[str]) -> list[str]:
        found: list[str] = []
        for canonical, pattern in PAYMENT_MODE_PATTERNS:
            for line in lines:
                if pattern.search(line):
                    found.append(canonical)
                    break
        return found

    def _contains_payment_term_hint(self, text: str) -> bool:
        lowered = text.lower()
        return any(k.lower() in lowered for k in PAYMENT_TERM_KEYWORDS)

    def _contains_premium_hint(self, text: str) -> bool:
        lowered = text.lower()
        return any(k.lower() in lowered for k in PREMIUM_HINT_KEYWORDS)

    def _extract_numeric_from_money_text(self, text: str) -> float | None:
        m = re.search(r"[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?", text or "")
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None

    def _extract_unit_from_money_text(self, text: str) -> str | None:
        for unit in ("HKD", "USD", "RMB", "CNY", "\u6e2f\u5143", "\u7f8e\u5143", "\u4eba\u6c11\u5e01"):
            if unit.lower() in (text or "").lower():
                return unit
        if "HK$" in (text or ""):
            return "HKD"
        if "US$" in (text or ""):
            return "USD"
        return None

    def _append_plan_level_premium_summary(self, facts: list[FactRecord]) -> list[FactRecord]:
        premium_facts = [f for f in facts if f.dimension_key == "premium_payment"]
        if not premium_facts:
            return facts

        all_annual_premiums: list[str] = []
        all_years: set[int] = set()
        all_modes: list[str] = []
        for fact in premium_facts:
            metadata = fact.metadata_json or {}
            premiums = metadata.get("annual_premium_values")
            if isinstance(premiums, list):
                for p in premiums:
                    if isinstance(p, str) and p and p not in all_annual_premiums:
                        all_annual_premiums.append(p)
            years = metadata.get("payment_terms_years")
            if isinstance(years, list):
                for value in years:
                    try:
                        year = int(value)
                    except (TypeError, ValueError):
                        continue
                    if 5 <= year <= 40:
                        all_years.add(year)

            modes = metadata.get("payment_modes")
            if isinstance(modes, list):
                for mode in modes:
                    if isinstance(mode, str) and mode not in all_modes:
                        all_modes.append(mode)

        if not all_annual_premiums and not all_years and not all_modes:
            return facts

        years = sorted(all_years)
        value_parts: list[str] = []
        normalized_parts: list[str] = []
        if all_annual_premiums:
            value_parts.append("\u5e74\u5ea6\u4fdd\u8d39\uff1a" + " / ".join(all_annual_premiums[:3]))
            normalized_parts.append("annual_premium_values:" + ",".join(all_annual_premiums[:3]))
        if all_modes:
            value_parts.append("\u7f34\u8d39\u65b9\u5f0f\uff1a" + " / ".join(all_modes))
            normalized_parts.append("payment_modes:" + ",".join(all_modes))
        if years and not all_annual_premiums:
            value_parts.append("\u4fdd\u8d39\u7f34\u4ed8\u671f\uff1a" + " / ".join(f"{y}\u5e74" for y in years))
            normalized_parts.append("payment_terms_years:" + ",".join(str(y) for y in years))

        anchor = sorted(
            premium_facts,
            key=lambda x: (x.confidence, -(x.source_page or 0)),
            reverse=True,
        )[0]

        facts.append(
            FactRecord(
                dimension_key="premium_payment",
                dimension_label=dimension_label("premium_payment"),
                value_text="\uff1b".join(value_parts),
                normalized_value="|".join(normalized_parts),
                numeric_value=(
                    self._extract_numeric_from_money_text(all_annual_premiums[0])
                    if all_annual_premiums
                    else (float(years[0]) if years else None)
                ),
                unit=(
                    self._extract_unit_from_money_text(all_annual_premiums[0])
                    if all_annual_premiums
                    else ("years" if years else None)
                ),
                condition_text=None,
                confidence=0.995 if all_annual_premiums else (0.995 if len(years) >= 2 else 0.9),
                source_page=anchor.source_page,
                source_section=anchor.source_section,
                source_quote=anchor.source_quote,
                metadata_json={
                    "annual_premium_values": all_annual_premiums[:3],
                    "payment_terms_years": years,
                    "payment_modes": all_modes,
                    "derived_from": "plan_aggregate",
                },
            )
        )
        return facts

    def _append_dimension_list_aggregates(self, facts: list[FactRecord]) -> list[FactRecord]:
        target_keys = {d.key for d in DIMENSIONS if d.key != "premium_payment"}
        for dim_key in target_keys:
            dim_facts = [f for f in facts if f.dimension_key == dim_key and (f.value_text or "").strip()]
            if len(dim_facts) < 2:
                continue

            ranked = sorted(
                dim_facts,
                key=lambda x: (x.confidence, -len((x.value_text or "").strip())),
                reverse=True,
            )
            values: list[str] = []
            normalized_values: list[str] = []
            for row in ranked:
                value = (row.value_text or "").strip()
                if not value:
                    continue
                norm = (row.normalized_value or value).strip().lower()
                if norm in normalized_values:
                    continue
                values.append(value[:160])
                normalized_values.append(norm)
                if len(values) >= 4:
                    break

            if len(values) < 2:
                continue

            anchor = ranked[0]
            agg_value = "\uff1b".join(values)
            facts.append(
                FactRecord(
                    dimension_key=dim_key,
                    dimension_label=dimension_label(dim_key),
                    value_text=agg_value,
                    normalized_value="|".join(normalized_values),
                    numeric_value=anchor.numeric_value,
                    unit=anchor.unit,
                    condition_text=None,
                    confidence=min(0.97, max(0.86, anchor.confidence + 0.08)),
                    source_page=anchor.source_page,
                    source_section=anchor.source_section,
                    source_quote=anchor.source_quote,
                    metadata_json={
                        "value_list": values,
                        "derived_from": "plan_dimension_group",
                    },
                )
            )
        return facts

    def _extract_numeric(self, text: str) -> tuple[float | None, str | None]:
        m = MONEY_PATTERN.search(text)
        if not m:
            return None, None
        raw_num = (m.group(1) or "").replace(",", "").strip()
        try:
            return float(raw_num), (m.group(2) or None)
        except ValueError:
            return None, (m.group(2) or None)

    def _dedup(self, facts: list[FactRecord]) -> list[FactRecord]:
        best_by_key: dict[tuple[str, str, int | None], FactRecord] = {}
        key_order: list[tuple[str, str, int | None]] = []
        for fact in facts:
            key = (fact.dimension_key, fact.value_text, fact.source_page)
            current = best_by_key.get(key)
            if current is None:
                best_by_key[key] = fact
                key_order.append(key)
                continue
            if fact.confidence > current.confidence:
                best_by_key[key] = fact
        return [best_by_key[k] for k in key_order]
