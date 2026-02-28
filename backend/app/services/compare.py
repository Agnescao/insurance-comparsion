from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import PolicyFact
from app.schemas import CellValue, CompareResponse, CompareRow, SourceRef
from app.services.dimensions import DEFAULT_DIMENSIONS, dimension_label


class CompareService:
    def build_compare(
        self,
        db: Session,
        plan_ids: list[str],
        dimensions: list[str] | None = None,
        filters: dict | None = None,
    ) -> CompareResponse:
        del filters  # reserved for future filtering logic

        dimension_keys = dimensions or DEFAULT_DIMENSIONS
        rows: list[CompareRow] = []

        for dim in dimension_keys:
            values: dict[str, CellValue] = {}
            normalized_values: list[str] = []
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

                best = facts[0]
                values[plan_id] = CellValue(
                    value=best.value_text,
                    confidence=float(best.confidence),
                    source=SourceRef(page=best.source_page, section=best.source_section, quote=best.source_quote),
                )
                norm = (best.normalized_value or best.value_text).strip().lower()
                if norm:
                    normalized_values.append(norm)

            is_diff = len(set(normalized_values)) > 1
            rows.append(
                CompareRow(
                    dimension_key=dim,
                    dimension_label=dimension_label(dim),
                    is_different=is_diff,
                    plan_values=values,
                )
            )

        return CompareResponse(
            generated_at=datetime.utcnow(),
            plan_ids=plan_ids,
            dimensions=dimension_keys,
            rows=rows,
        )
