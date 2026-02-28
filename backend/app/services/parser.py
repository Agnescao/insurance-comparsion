from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz


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

            markdown = "\n".join(lines)
            pages.append(
                PageDocument(
                    page_number=page_idx + 1,
                    markdown=markdown,
                    layout={"page": page_idx + 1, "blocks": layout_dict.get("blocks", [])},
                )
            )

        plan_name = pdf_path.stem
        if pages and pages[0].markdown:
            head = pages[0].markdown.splitlines()[0].strip()
            if 2 <= len(head) <= 80:
                plan_name = head

        return ParsedPolicy(plan_name=plan_name, source_file=str(pdf_path), pages=pages)
