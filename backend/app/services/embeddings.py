from __future__ import annotations

import hashlib
import math
import time
from abc import ABC, abstractmethod

import requests

from app.config import settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in text.split():
                digest = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class QwenEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        url: str,
        *,
        timeout_sec: int = 30,
        max_retries: int = 4,
        retry_backoff_sec: float = 1.2,
        batch_size: int = 8,
        trust_env_proxy: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.timeout_sec = int(timeout_sec)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self.batch_size = max(1, int(batch_size))
        self.session = requests.Session()
        # Keep false by default to avoid stale HTTP(S)_PROXY causing connection resets.
        self.session.trust_env = bool(trust_env_proxy)

    def _embed_batch(self, batch: list[str], headers: dict[str, str]) -> list[list[float]]:
        payload = {"model": self.model, "input": {"texts": batch}}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(self.url, headers=headers, json=payload, timeout=self.timeout_sec)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"transient status={response.status_code}, body={response.text[:200]}",
                        response=response,
                    )
                response.raise_for_status()

                data = response.json()
                rows = data["output"]["embeddings"]
                vectors = [[float(x) for x in row["embedding"]] for row in rows]
                if len(vectors) != len(batch):
                    raise ValueError(f"Qwen embedding size mismatch: expected {len(batch)}, got {len(vectors)}")
                return vectors
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.retry_backoff_sec * (2**attempt)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        assert last_error is not None
        raise RuntimeError(
            "Qwen embedding request failed after retries. "
            "If you use local proxy tooling, check HTTP_PROXY/HTTPS_PROXY or set qwen_trust_env_proxy=true."
        ) from last_error

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors.extend(self._embed_batch(batch, headers))
        return vectors


def build_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "qwen" and settings.qwen_api_key:
        return QwenEmbeddingProvider(
            api_key=settings.qwen_api_key,
            model=settings.qwen_embedding_model,
            url=settings.qwen_embedding_url,
            timeout_sec=settings.qwen_timeout_sec,
            max_retries=settings.qwen_max_retries,
            retry_backoff_sec=settings.qwen_retry_backoff_sec,
            batch_size=settings.qwen_batch_size,
            trust_env_proxy=settings.qwen_trust_env_proxy,
        )
    return HashEmbeddingProvider(dim=settings.embedding_dim)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1)) or 1.0
    n2 = math.sqrt(sum(b * b for b in v2)) or 1.0
    return dot / (n1 * n2)
