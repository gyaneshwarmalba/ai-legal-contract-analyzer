"""Vector stores for clause retrieval.

Both stores use inner product over L2-normalised vectors (cosine similarity).
`FaissVectorStore` is a drop-in for larger corpora; `InMemoryVectorStore`
needs only numpy.
"""

from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from app.models import Clause, RetrievedClause


class VectorStore(Protocol):
    def add(self, clauses: list[Clause], vectors: np.ndarray) -> None: ...

    def remove_contract(self, contract_id: str) -> None: ...

    def search(self, query: np.ndarray, k: int, contract_id: str | None = None) -> list[RetrievedClause]: ...

    def __len__(self) -> int: ...


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._clauses: list[Clause] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.Lock()

    def add(self, clauses: list[Clause], vectors: np.ndarray) -> None:
        if vectors.shape != (len(clauses), self.dim):
            raise ValueError(f"expected vectors of shape {(len(clauses), self.dim)}, got {vectors.shape}")
        with self._lock:
            self._clauses.extend(clauses)
            self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])

    def remove_contract(self, contract_id: str) -> None:
        with self._lock:
            keep = [i for i, c in enumerate(self._clauses) if c.contract_id != contract_id]
            self._clauses = [self._clauses[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)

    def search(self, query: np.ndarray, k: int, contract_id: str | None = None) -> list[RetrievedClause]:
        with self._lock:
            if not self._clauses:
                return []
            scores = self._matrix @ query.astype(np.float32).ravel()
            order = np.argsort(-scores)
            results = []
            for idx in order:
                clause = self._clauses[idx]
                if contract_id and clause.contract_id != contract_id:
                    continue
                results.append(RetrievedClause(clause=clause, score=float(scores[idx])))
                if len(results) == k:
                    break
            return results

    def __len__(self) -> int:
        return len(self._clauses)


class FaissVectorStore:
    """FAISS flat inner-product index. Rebuilt on removal; fine for tens of thousands of clauses."""

    def __init__(self, dim: int) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("VECTOR_STORE=faiss requires faiss-cpu (pip install -r requirements-cloud.txt)") from exc
        self._faiss = faiss
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._clauses: list[Clause] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.Lock()

    def add(self, clauses: list[Clause], vectors: np.ndarray) -> None:
        with self._lock:
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)
            self._index.add(vecs)
            self._clauses.extend(clauses)
            self._vectors = np.vstack([self._vectors, vecs])

    def remove_contract(self, contract_id: str) -> None:
        with self._lock:
            keep = [i for i, c in enumerate(self._clauses) if c.contract_id != contract_id]
            self._clauses = [self._clauses[i] for i in keep]
            self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
            self._index = self._faiss.IndexFlatIP(self.dim)
            if keep:
                self._index.add(self._vectors)

    def search(self, query: np.ndarray, k: int, contract_id: str | None = None) -> list[RetrievedClause]:
        with self._lock:
            if not self._clauses:
                return []
            fetch = len(self._clauses) if contract_id else min(k, len(self._clauses))
            scores, ids = self._index.search(np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), fetch)
            results = []
            for score, idx in zip(scores[0], ids[0], strict=False):
                if idx < 0:
                    continue
                clause = self._clauses[idx]
                if contract_id and clause.contract_id != contract_id:
                    continue
                results.append(RetrievedClause(clause=clause, score=float(score)))
                if len(results) == k:
                    break
            return results

    def __len__(self) -> int:
        return len(self._clauses)
