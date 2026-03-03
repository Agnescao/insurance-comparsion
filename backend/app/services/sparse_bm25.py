from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def term_index(token: str, buckets: int = 2_000_003) -> int:
    # Use deterministic hashing so ingest and query map to identical sparse dims
    # across different Python processes and machines.
    digest = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16) % buckets


class BM25SparseEncoder:
    """Build BM25-style sparse vectors for documents and queries."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = max(1, len(docs_tokens))
        self.avgdl = sum(len(toks) for toks in docs_tokens) / self.doc_count if docs_tokens else 1.0

        self.df: dict[str, int] = defaultdict(int)
        for toks in docs_tokens:
            for t in set(toks):
                self.df[t] += 1

    def idf(self, token: str) -> float:
        n_t = self.df.get(token, 0)
        return math.log(1.0 + (self.doc_count - n_t + 0.5) / (n_t + 0.5))

    def encode_doc(self, tokens: list[str]) -> dict[int, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        dl = len(tokens)
        out: dict[int, float] = {}
        for t, f in tf.items():
            idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            w = idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
            if w > 0:
                idx = term_index(t)
                out[idx] = out.get(idx, 0.0) + float(w)
        return out

    def encode_query(self, tokens: list[str]) -> dict[int, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        out: dict[int, float] = {}
        for t, f in tf.items():
            idf = self.idf(t)
            w = idf * float(f)
            if w > 0:
                idx = term_index(t)
                out[idx] = out.get(idx, 0.0) + float(w)
        return out
