from __future__ import annotations

import hashlib
import math
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
    def __init__(self, api_key: str, model: str, url: str):
        self.api_key = api_key
        self.model = model
        self.url = url

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for text in texts:
            payload = {"model": self.model, "input": {"texts": [text]}}
            response = requests.post(self.url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            emb = data["output"]["embeddings"][0]["embedding"]
            vectors.append([float(x) for x in emb])
        return vectors


def build_embedding_provider() -> EmbeddingProvider:
    if settings.embedding_provider.lower() == "qwen" and settings.qwen_api_key:
        return QwenEmbeddingProvider(
            api_key=settings.qwen_api_key,
            model=settings.qwen_embedding_model,
            url=settings.qwen_embedding_url,
        )
    return HashEmbeddingProvider(dim=settings.embedding_dim)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1)) or 1.0
    n2 = math.sqrt(sum(b * b for b in v2)) or 1.0
    return dot / (n1 * n2)
