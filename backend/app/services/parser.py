from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
import warnings

from app.config import settings

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except Exception:
    import fitz  # fallback for older PyMuPDF installs

if not hasattr(fitz, "open"):
    raise RuntimeError(
        "Invalid 'fitz' module detected (missing open). "
        "Please uninstall package 'fitz' and install 'pymupdf'."
    )

try:
    from opencc import OpenCC
except Exception:  # pragma: no cover - optional dependency
    OpenCC = None


@dataclass
class PageDocument:
    page_number: int
    markdown: str
    layout: dict[str, Any]


@dataclass
class ParsedPolicy:
    plan_name: str
    source_file: str
    pages: list[PageDocument]


class PDFParser:
    """Parse PDF into page-level markdown + layout json.

    In production this can be wired to online OCR/layout models. For local prototype,
    PyMuPDF extraction is used as reliable fallback.
    """

    def __init__(self) -> None:
        self.normalize_to_simplified = bool(settings.normalize_to_simplified)
        self._cc = None
        if self.normalize_to_simplified and OpenCC is not None:
            try:
                self._cc = OpenCC("t2s")
            except Exception:
                warnings.warn(
                    "normalize_to_simplified=True but OpenCC runtime config is unavailable; text will keep original script.",
                    RuntimeWarning,
                )
                self._cc = None
        elif self.normalize_to_simplified and OpenCC is None:
            warnings.warn(
                "normalize_to_simplified=True but opencc is not installed; text will keep original script.",
                RuntimeWarning,
            )

    def _normalize_text(self, text: str) -> str:
        value = text or ""
        if self._cc:
            value = self._cc.convert(value)
        return value

    def parse(self, pdf_path: Path) -> ParsedPolicy:
        doc = fitz.open(pdf_path)
        pages: list[PageDocument] = []

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            layout_dict = page.get_text("dict")
            lines: list[str] = []
            for block in layout_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    parts: list[str] = []
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            parts.append(text)
                    if parts:
                        lines.append(" ".join(parts))

            markdown = self._normalize_text("\n".join(lines))
            pages.append(
                PageDocument(
                    page_number=page_idx + 1,
                    markdown=markdown,
                    layout={"page": page_idx + 1, "blocks": layout_dict.get("blocks", [])},
                )
            )

        plan_name = self._extract_plan_name(pages, pdf_path)

        return ParsedPolicy(
            plan_name=self._normalize_text(plan_name),
            source_file=str(pdf_path),
            pages=pages,
        )

    def _extract_plan_name(self, pages: list[PageDocument], pdf_path: Path) -> str:
        if not pages:
            return pdf_path.stem

        lines: list[str] = []
        for page in pages[:2]:
            for line in page.markdown.splitlines():
                clean = " ".join(line.split()).strip()
                if clean:
                    lines.append(clean)
            if len(lines) >= 120:
                break

        if not lines:
            return pdf_path.stem

        generic_titles = {
            "閱覽電子版",
            "阅览电子版",
            "閱 覽 電 子 版",
            "AIA International Limited",
            "友邦保險(國際)有限公司",
            "友邦保险(国际)有限公司",
            "友邦保險（國際）有限公司",
            "友邦保险（国际）有限公司",
        }

        preferred_keywords = (
            "保險計劃",
            "保险计划",
            "醫療計劃",
            "医疗计划",
            "危疾",
            "明珠",
            "Plan",
            "Medical",
        )

        stop_phrases = (
            "本公司",
            "我們",
            "我们",
            "是指",
            "分別指",
            "分别指",
            "行政區",
            "行政区",
            "於",
            "于",
            "有限公司",
            "AIA",
            "友邦",
        )

        # 1) Prefer short quoted product titles on cover page.
        for line in lines[:25]:
            if line in generic_titles:
                continue
            if any(x in line for x in stop_phrases):
                continue
            if "「" in line and "」" in line and 2 <= len(line) <= 36:
                return line

        # 2) Prefer keyword-bearing title lines, but avoid disclaimer/sentence-like strings.
        keyword_candidates: list[str] = []
        for line in lines:
            if line in generic_titles:
                continue
            if any(x in line for x in stop_phrases):
                continue
            if not (2 <= len(line) <= 48):
                continue
            if line.endswith(("。", ".", "，", ",")):
                continue
            if any(k.lower() in line.lower() for k in preferred_keywords):
                keyword_candidates.append(line)
        if keyword_candidates:
            return sorted(keyword_candidates, key=lambda x: (len(x), x))[0]

        # 3) Fallback to sanitized filename when no reliable title is found.
        return self._from_filename(pdf_path.stem)

    def _from_filename(self, stem: str) -> str:
        text = stem
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(r"\s*\(\d+\)\s*$", "", text)
        text = re.sub(r"\bsc\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return stem
        if text.isascii():
            return text.title()
        return text
