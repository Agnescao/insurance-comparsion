from __future__ import annotations

import logging
import re
from datetime import datetime
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PolicyFact
from app.schemas import CellValue, CompareResponse, CompareRow, SourceRef
from app.services.dimensions import DEFAULT_DIMENSIONS, dimension_label


class CompareService:
    def __init__(self) -> None:
        self.logger = logging.getLogger("uvicorn.error")

    def build_compare(
        self,
        db: Session,
        plan_ids: list[str],
        dimensions: list[str] | None = None,
        filters: dict | None = None,
    ) -> CompareResponse:
        del filters  # reserved for future filtering logic

        dimension_keys = dimensions or DEFAULT_DIMENSIONS
        t0 = perf_counter()
        self.logger.info("compare.build.start plans=%s dims=%s backend=sqlite_facts", plan_ids, dimension_keys)

        rows: list[CompareRow] = []
        for dim in dimension_keys:
            dt0 = perf_counter()
            values: dict[str, CellValue] = {}
            normalized_values: list[str] = []
            hit_count = 0

            for plan_id in plan_ids:
                facts = db.execute(
                    select(PolicyFact)
                    .where(PolicyFact.plan_id == plan_id, PolicyFact.dimension_key == dim)
                    .order_by(PolicyFact.confidence.desc(), PolicyFact.created_at.desc())
                ).scalars().all()

                if not facts:
                    values[plan_id] = CellValue(
                        value="未提取到",
                        confidence=0.0,
                        source=SourceRef(page=None, section=None, quote=None),
                    )
                    continue

                hit_count += 1
                best = facts[0]
                value_text = best.value_text
                source_quote = best.source_quote
                confidence = float(best.confidence)
                source_page = best.source_page
                source_section = best.source_section

                if dim == "coverage_surgery":
                    value_text = self._summarize_surgery_facts(facts)
                    source_quote = self._shorten(best.source_quote or best.value_text or "", 240)
                    # Summary is derived from multiple facts; keep confidence conservative.
                    confidence = min(0.92, max(0.55, float(best.confidence)))

                values[plan_id] = CellValue(
                    value=value_text,
                    confidence=confidence,
                    source=SourceRef(page=source_page, section=source_section, quote=source_quote),
                )
                norm = (best.normalized_value or value_text).strip().lower()
                if norm and norm != "未提取到":
                    normalized_values.append(norm)

            is_diff = len(set(normalized_values)) > 1 if normalized_values else False
            rows.append(
                CompareRow(
                    dimension_key=dim,
                    dimension_label=dimension_label(dim),
                    is_different=is_diff,
                    plan_values=values,
                )
            )
            self.logger.info(
                "compare.dimension.done dim=%s hit_plans=%d/%d is_diff=%s elapsed=%.3fs",
                dim,
                hit_count,
                len(plan_ids),
                is_diff,
                perf_counter() - dt0,
            )

        self.logger.info("compare.build.done row_count=%d elapsed=%.3fs", len(rows), perf_counter() - t0)
        return CompareResponse(
            generated_at=datetime.utcnow(),
            plan_ids=plan_ids,
            dimensions=dimension_keys,
            rows=rows,
        )

    def _summarize_surgery_facts(self, facts: list[PolicyFact]) -> str:
        snippets: list[str] = []
        for fact in facts[:8]:
            snippets.extend(self._extract_surgery_snippets(fact.value_text or ""))
            if fact.source_quote:
                snippets.extend(self._extract_surgery_snippets(fact.source_quote or ""))

        deduped = list(dict.fromkeys([s for s in snippets if s]))
        if not deduped:
            return self._shorten((facts[0].value_text if facts else "未提取到") or "未提取到", 180)

        ranked_items = sorted(deduped, key=self._score_surgery_item, reverse=True)
        top_items = ranked_items[:6]
        categories = self._surgery_categories(top_items)
        cat_labels = [label for label, _ in categories[:3]]
        concrete_items = [item for item in top_items if not self._is_generic_surgery_item(item)]
        item_labels = concrete_items[:3]

        parts: list[str] = []
        if cat_labels:
            parts.append(f"主要手术类型: {'、'.join(cat_labels)}")
        if item_labels:
            parts.append(f"代表项目: {'；'.join(item_labels)}")
        else:
            parts.append("给付方式: 以手术复杂程度/分类表为准，未列出明确术式名称")
        return self._shorten("；".join(parts), 180)

    def _extract_surgery_snippets(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []

        parts = re.split(r"[；;。,.，\n]", cleaned)
        surgery_keywords = (
            "手术",
            "手術",
            "surgery",
            "operation",
            "切除",
            "移植",
            "搭桥",
            "搭橋",
            "介入",
            "成形术",
            "成形術",
            "植入",
            "置换",
            "置換",
            "修补",
            "修補",
            "ectomy",
            "otomy",
        )
        generic_noise = (
            "复杂的手术",
            "複雜的手術",
            "按自愿医保计划",
            "按自願醫保計劃",
            "指定期间",
            "指定期間",
        )
        snippets: list[str] = []
        for raw in parts:
            seg = re.sub(r"^[0-9A-Za-z一二三四五六七八九十\-\)\.(（(]+", "", raw.strip())
            if len(seg) < 2:
                continue
            lowered = seg.lower()
            if any(k.lower() in lowered for k in surgery_keywords):
                if any(n.lower() in lowered for n in generic_noise) and len(seg) < 18:
                    continue
                snippets.append(self._shorten(seg, 42))

        if snippets:
            return snippets
        return [self._shorten(cleaned, 42)]

    def _surgery_categories(self, snippets: list[str]) -> list[tuple[str, int]]:
        rules: list[tuple[str, tuple[str, ...]]] = [
            ("癌症相关", ("癌", "肿瘤", "腫瘤", "化疗", "化療", "放疗", "放療", "标靶", "標靶")),
            ("心血管", ("心", "冠状动脉", "冠狀動脈", "瓣膜", "主动脉", "主動脈", "血管")),
            ("器官移植", ("移植", "器官", "骨髓")),
            ("消化系统", ("结肠", "結腸", "肠", "腸", "胃", "肝", "胆", "膽", "胰")),
            ("神经系统", ("脑", "腦", "脊髓", "颈动脉", "頸動脈")),
            ("日间/门诊", ("日间", "日間", "门诊", "門診")),
            ("复杂/大型手术", ("复杂", "複雜", "大型", "深切治疗", "深切治療", "介入", "搭桥", "搭橋")),
        ]

        scores: list[tuple[str, int]] = []
        lowered_snippets = [s.lower() for s in snippets]
        for label, keywords in rules:
            count = 0
            for seg in lowered_snippets:
                if any(k.lower() in seg for k in keywords):
                    count += 1
            if count > 0:
                scores.append((label, count))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _score_surgery_item(self, text: str) -> int:
        lowered = (text or "").lower()
        score = 0
        specific_terms = (
            "切除",
            "移植",
            "搭桥",
            "搭橋",
            "介入",
            "植入",
            "成形术",
            "成形術",
            "置换",
            "修补",
            "心包",
            "冠状动脉",
            "冠狀動脈",
            "颈动脉",
            "頸動脈",
        )
        generic_terms = ("复杂的手术", "複雜的手術", "相关手术", "住院及手术")
        if any(t.lower() in lowered for t in specific_terms):
            score += 4
        if any(t.lower() in lowered for t in generic_terms):
            score -= 3
        length = len(text or "")
        if 8 <= length <= 36:
            score += 2
        elif length > 55:
            score -= 1
        return score

    def _is_generic_surgery_item(self, text: str) -> bool:
        lowered = (text or "").lower()
        specific_terms = (
            "切除",
            "移植",
            "搭桥",
            "搭橋",
            "介入",
            "植入",
            "成形术",
            "成形術",
            "置换",
            "置換",
            "修补",
            "修補",
            "ectomy",
            "otomy",
        )
        if any(term.lower() in lowered for term in specific_terms):
            return False
        generic_terms = (
            "复杂的手术",
            "複雜的手術",
            "复杂手术",
            "複雜手術",
            "手术表分类",
            "手術表分類",
            "按自愿医保计划",
            "按自願醫保計劃",
            "相关手术",
            "住院及手术",
        )
        return any(term.lower() in lowered for term in generic_terms)

    def _shorten(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "..."
