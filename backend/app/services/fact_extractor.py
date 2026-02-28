from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.dimensions import DIMENSIONS, dimension_label
from app.services.chunking import ChunkDoc


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


MONEY_PATTERN = re.compile(r"([\$¥￥]?\s?\d{1,3}(?:[,\d]{0,12})(?:\.\d+)?)\s*(元|人民币|HKD|USD|RMB|%|次|天|年)?")


class FactExtractor:
    def __init__(self) -> None:
        self.dimension_patterns: dict[str, list[re.Pattern[str]]] = {}
        for d in DIMENSIONS:
            pats = [re.compile(re.escape(k), re.IGNORECASE) for k in d.keywords if k]
            self.dimension_patterns[d.key] = pats

    def extract_from_chunks(self, chunks: list[ChunkDoc]) -> list[FactRecord]:
        facts: list[FactRecord] = []
        for chunk in chunks:
            lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
            if not lines:
                continue
            for dim in DIMENSIONS:
                matched_lines = [line for line in lines if self._line_match(line, dim.key)]
                if not matched_lines:
                    continue
                sample = matched_lines[0]
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

            # disease-specific scenario as dynamic dimension
            for disease, key in (("卵巢癌", "condition_ovarian_cancer"), ("cancer", "condition_cancer")):
                disease_lines = [line for line in lines if disease.lower() in line.lower()]
                if disease_lines:
                    sample = disease_lines[0]
                    num, unit = self._extract_numeric(sample)
                    facts.append(
                        FactRecord(
                            dimension_key=key,
                            dimension_label=dimension_label(key),
                            value_text=sample,
                            normalized_value=sample.lower(),
                            numeric_value=num,
                            unit=unit,
                            condition_text=disease,
                            confidence=0.58,
                            source_page=chunk.page_start,
                            source_section=chunk.section_path,
                            source_quote=sample[:300],
                        )
                    )

        return self._dedup(facts)

    def _line_match(self, line: str, dimension_key: str) -> bool:
        patterns = self.dimension_patterns.get(dimension_key, [])
        return any(pattern.search(line) for pattern in patterns)

    def _extract_numeric(self, text: str) -> tuple[float | None, str | None]:
        m = MONEY_PATTERN.search(text)
        if not m:
            return None, None
        raw_num = m.group(1).replace(",", "").replace(" ", "").replace("$", "").replace("¥", "").replace("￥", "")
        try:
            return float(raw_num), (m.group(2) or None)
        except ValueError:
            return None, (m.group(2) or None)

    def _dedup(self, facts: list[FactRecord]) -> list[FactRecord]:
        seen: set[tuple[str, str, int | None]] = set()
        out: list[FactRecord] = []
        for fact in facts:
            key = (fact.dimension_key, fact.value_text, fact.source_page)
            if key in seen:
                continue
            seen.add(key)
            out.append(fact)
        return out
