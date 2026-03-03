from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionDef:
    key: str
    label: str
    keywords: tuple[str, ...]


DIMENSIONS: tuple[DimensionDef, ...] = (
    DimensionDef(
        "coverage_hospitalization",
        "\u4f4f\u9662\u4fdd\u969c",
        ("\u4f4f\u9662", "\u4f4f\u9662", "\u4f4f\u9662\u75c5\u623f", "\u4f4f\u9662\u8cbb\u7528", "hospital", "inpatient"),
    ),
    DimensionDef(
        "coverage_outpatient",
        "\u95e8\u8bca\u4fdd\u969c",
        ("\u95e8\u8bca", "\u9580\u8a3a", "\u95e8\u8a3a", "outpatient", "clinic"),
    ),
    DimensionDef(
        "coverage_surgery",
        "\u624b\u672f\u4fdd\u969c",
        ("\u624b\u672f", "\u624b\u8853", "surgery", "operation"),
    ),
    DimensionDef(
        "premium_payment",
        "\u4fdd\u8d39\u4e0e\u652f\u4ed8\u65b9\u5f0f",
        (
            "\u4fdd\u8d39",
            "\u4fdd\u8cbb",
            "\u7f34\u8d39",
            "\u7e73\u8cbb",
            "\u7f34\u4ed8",
            "\u7e73\u4ed8",
            "\u4ed8\u8d39",
            "\u4ed8\u6b3e\u65b9\u5f0f",
            "payment",
            "premium",
            "\u5206\u671f",
            "\u5e74\u7f34",
            "\u534a\u5e74\u7f34",
            "\u5b63\u7f34",
            "\u6708\u7f34",
        ),
    ),
    DimensionDef(
        "annual_limit",
        "\u5e74\u5ea6\u9650\u989d",
        (
            "\u5e74\u5ea6\u9650\u989d",
            "\u6bcf\u5e74\u9650\u989d",
            "annual limit",
            "\u6bcf\u5e74",
            "\u9650\u989d",
            "\u603b\u8d54\u4ed8",
            "\u4e2a\u4eba\u6700\u9ad8\u8d54\u507f\u9650\u989d",
            "\u500b\u4eba\u6700\u9ad8\u8ce0\u511f\u9650\u984d",
            "\u6700\u9ad8\u8d54\u507f\u9650\u989d",
            "\u6700\u9ad8\u8ce0\u511f\u9650\u984d",
            "\u6bcf\u4fdd\u5355\u53ef\u83b7\u652f\u4ed8",
        ),
    ),
    DimensionDef(
        "itemized_limit",
        "\u5206\u9879\u9650\u989d",
        (
            "\u5206\u9879",
            "\u5206\u9805",
            "\u5b50\u9650\u989d",
            "itemized",
            "sub-limit",
            "\u4e2a\u4eba\u6700\u9ad8\u8d54\u507f\u9650\u989d",
            "\u500b\u4eba\u6700\u9ad8\u8ce0\u511f\u9650\u984d",
            "\u4e2a\u4eba\u6700\u9ad8\u8d54\u507f\u9650\u989d 50,000",
            "\u500b\u4eba\u6700\u9ad8\u8ce0\u511f\u9650\u984d 50,000",
            "\u6bcf\u9879\u8d54\u507f\u4e0a\u9650",
            "\u6bcf\u9805\u8ce0\u511f\u4e0a\u9650",
        ),
    ),
    DimensionDef(
        "deductible_copay",
        "\u81ea\u4ed8\u989d\u4e0e\u5171\u4ed8",
        ("\u81ea\u4ed8", "\u81ea\u8ca0", "\u514d\u8d54", "\u514d\u8ce0", "\u5171\u4ed8", "\u5171\u8ca0", "deductible", "copay"),
    ),
    DimensionDef(
        "riders_benefits",
        "\u9644\u52a0\u9669\u4e0e\u9644\u52a0\u798f\u5229",
        ("\u9644\u52a0", "\u9644\u52a0\u5951\u7d04", "rider", "benefit", "\u989d\u5916\u4fdd\u969c", "\u589e\u503c\u670d\u52a1"),
    ),
    DimensionDef(
        "exclusions",
        "\u9664\u5916\u8d23\u4efb\u4e0e\u7279\u6b8a\u6761\u4ef6",
        ("\u9664\u5916", "\u9664\u5916\u8cac\u4efb", "\u4e0d\u4fdd", "exclusion", "\u514d\u8d23", "\u514d\u8cac", "\u7b49\u5f85\u671f"),
    ),
)

DEFAULT_DIMENSIONS: list[str] = [d.key for d in DIMENSIONS]

CONDITION_TERMS: dict[str, tuple[str, ...]] = {
    "condition_ovarian_cancer": ("\u5375\u5de2\u764c", "\u5375\u5de2\u60e1\u6027\u816b\u7624", "ovarian cancer"),
    "condition_heart_disease": ("\u5fc3\u810f\u75c5", "\u5fc3\u81df\u75c5", "heart disease", "cardiac"),
    "condition_cancer": ("\u764c\u75c7", "\u6076\u6027\u80bf\u7624", "cancer"),
}


def dimension_label(key: str) -> str:
    for d in DIMENSIONS:
        if d.key == key:
            return d.label
    if key.startswith("condition_"):
        return f"\u75be\u75c5\u573a\u666f: {key.removeprefix('condition_').replace('_', ' ')}"
    return key


def detect_dimensions(text: str) -> list[str]:
    lowered = (text or "").lower()
    found: list[str] = []
    for d in DIMENSIONS:
        if any(k.lower() in lowered for k in d.keywords):
            found.append(d.key)

    for dim_key, terms in CONDITION_TERMS.items():
        if any(t.lower() in lowered for t in terms):
            found.append(dim_key)

    return list(dict.fromkeys(found))


def condition_dimension_for_query(text: str) -> str | None:
    lowered = (text or "").lower()
    for dim_key, terms in CONDITION_TERMS.items():
        if any(t.lower() in lowered for t in terms):
            return dim_key
    return None


def all_dimensions() -> list[dict[str, str]]:
    return [{"key": d.key, "label": d.label} for d in DIMENSIONS]
