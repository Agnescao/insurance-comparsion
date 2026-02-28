from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionDef:
    key: str
    label: str
    keywords: tuple[str, ...]


DIMENSIONS: tuple[DimensionDef, ...] = (
    DimensionDef("coverage_hospitalization", "住院保障", ("住院", "hospital", "inpatient", "住院保障")),
    DimensionDef("coverage_outpatient", "门诊保障", ("门诊", "outpatient", "clinic", "门急诊")),
    DimensionDef("coverage_surgery", "手术保障", ("手术", "surgery", "operation")),
    DimensionDef("premium_payment", "保费与支付方式", ("保费", "缴费", "payment", "premium", "分期")),
    DimensionDef("annual_limit", "年度限额", ("年度限额", "annual limit", "每年", "限额", "总赔付")),
    DimensionDef("itemized_limit", "分项限额", ("分项", "子限额", "itemized", "sub-limit")),
    DimensionDef("deductible_copay", "自付额与共付", ("自付", "免赔", "共付", "deductible", "copay")),
    DimensionDef("riders_benefits", "附加险与附加福利", ("附加", "rider", "benefit", "额外保障", "增值服务")),
    DimensionDef("exclusions", "除外责任与特殊条件", ("除外", "不保", "exclusion", "免责", "等待期")),
)

DEFAULT_DIMENSIONS: list[str] = [d.key for d in DIMENSIONS]


def dimension_label(key: str) -> str:
    for d in DIMENSIONS:
        if d.key == key:
            return d.label
    if key.startswith("condition_"):
        return f"疾病场景: {key.removeprefix('condition_').replace('_', ' ')}"
    return key


def detect_dimensions(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for d in DIMENSIONS:
        if any(k.lower() in lowered for k in d.keywords):
            found.append(d.key)
    if "卵巢癌" in text:
        found.append("condition_ovarian_cancer")
    # Preserve order while deduplicating.
    return list(dict.fromkeys(found))


def all_dimensions() -> list[dict[str, str]]:
    return [{"key": d.key, "label": d.label} for d in DIMENSIONS]
