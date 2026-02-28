from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.services.embeddings import EmbeddingProvider, cosine_similarity
from app.services.parser import ParsedPolicy


@dataclass
class ChunkDoc:
    text: str
    page_start: int
    page_end: int
    section_path: str | None
    paragraph_index: int
    embedding: list[float] | None


class HybridChunker:
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    def chunk_policy(self, parsed: ParsedPolicy) -> list[ChunkDoc]:
        raw_chunks: list[ChunkDoc] = []
        for page in parsed.pages:
            page_chunks = self._recursive_split(page.markdown)
            for idx, text in enumerate(page_chunks):
                raw_chunks.append(
                    ChunkDoc(
                        text=text,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        section_path=f"page-{page.page_number}",
                        paragraph_index=idx,
                        embedding=None,
                    )
                )

        if not raw_chunks:
            return []

        embeddings = self.embedding_provider.embed([c.text for c in raw_chunks])
        for c, e in zip(raw_chunks, embeddings):
            c.embedding = e

        return self._semantic_merge(raw_chunks)

    def _recursive_split(self, text: str) -> list[str]:
        clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not clean:
            return []

        size = settings.chunk_size
        overlap = settings.chunk_overlap
        if len(clean) <= size:
            return [clean]

        separators = ["\n\n", "\n", "。", ". ", "；", ";", "，", ",", " "]
        chunks = [clean]
        for sep in separators:
            next_chunks: list[str] = []
            for chunk in chunks:
                if len(chunk) <= size:
                    next_chunks.append(chunk)
                    continue
                parts = chunk.split(sep)
                if len(parts) == 1:
                    next_chunks.append(chunk)
                    continue

                buf = ""
                joiner = sep if sep not in [" "] else " "
                for part in parts:
                    candidate = f"{buf}{joiner if buf else ''}{part}" if part else buf
                    if len(candidate) <= size:
                        buf = candidate
                    else:
                        if buf:
                            next_chunks.append(buf)
                        buf = part
                if buf:
                    next_chunks.append(buf)
            chunks = next_chunks

        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= size:
                final_chunks.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                end = start + size
                final_chunks.append(chunk[start:end])
                if end >= len(chunk):
                    break
                start = max(0, end - overlap)

        return [c.strip() for c in final_chunks if c.strip()]

    def _semantic_merge(self, chunks: list[ChunkDoc]) -> list[ChunkDoc]:
        if not chunks:
            return []

        merged: list[ChunkDoc] = [chunks[0]]
        threshold = settings.semantic_merge_threshold

        for current in chunks[1:]:
            previous = merged[-1]
            sim = cosine_similarity(previous.embedding or [], current.embedding or [])
            can_merge = (
                sim >= threshold
                and previous.page_end + 1 >= current.page_start
                and len(previous.text) + len(current.text) + 1 <= int(settings.chunk_size * 1.35)
            )
            if can_merge:
                previous.text = f"{previous.text}\n{current.text}"
                previous.page_end = max(previous.page_end, current.page_end)
                if previous.embedding and current.embedding:
                    previous.embedding = [(a + b) / 2.0 for a, b in zip(previous.embedding, current.embedding)]
            else:
                merged.append(current)

        return merged
