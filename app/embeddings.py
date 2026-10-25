"""Embedding providers.

`HashingEmbedder` is a deterministic, dependency-light embedder (signed feature
hashing over word unigrams and bigrams) so the pipeline and its tests run with
no network access. `BedrockEmbedder` calls Amazon Titan Text Embeddings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or shall that the this to under was were will with "
    "any each either may such which who what when where how does do party parties agreement".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")


SUFFIXES = ("ation", "ated", "ate", "ment", "ing", "ed", "e")


def stem(tok: str) -> str:
    """Light suffix stripping so "obligations"/"obligation" and "terminate"/"termination" share a stem."""
    if tok.endswith("ies") and len(tok) > 4:
        tok = tok[:-3] + "y"
    elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 4:
        tok = tok[:-1]
    for suffix in SUFFIXES:
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(tok) for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS and len(tok) >= 2]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
            counts: dict[str, int] = {}
            for f in features:
                counts[f] = counts.get(f, 0) + 1
            for f, c in counts.items():
                idx, sign = self._bucket(f)
                out[row, idx] += sign * (1.0 + math.log(c))
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


class BedrockEmbedder:
    def __init__(self, model_id: str, region: str, dim: int = 1024) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.dim = dim
        self.name = model_id

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            try:
                resp = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps({"inputText": text[:8000], "dimensions": self.dim, "normalize": True}),
                )
            except Exception:
                logger.exception("bedrock embedding request failed", extra={"model_id": self.model_id})
                raise
            vectors.append(json.loads(resp["body"].read())["embedding"])
        return np.asarray(vectors, dtype=np.float32)
