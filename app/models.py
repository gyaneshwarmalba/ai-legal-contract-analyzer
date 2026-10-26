"""Domain models shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Clause:
    """A clause-aware chunk of a contract; the unit of retrieval and risk scoring."""

    contract_id: str
    clause_id: str
    number: str
    heading: str
    text: str

    @property
    def citation_label(self) -> str:
        return f"{self.contract_id} §{self.number} {self.heading}".strip()


@dataclass
class Contract:
    contract_id: str
    title: str
    source: str
    clauses: list[Clause] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "title": self.title,
            "source": self.source,
            "clause_count": len(self.clauses),
        }


@dataclass(frozen=True)
class RetrievedClause:
    clause: Clause
    score: float


@dataclass
class Citation:
    index: int
    contract_id: str
    clause_id: str
    label: str
    score: float
    excerpt: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    model: str
    needs_review: bool
    review_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskFinding:
    rule_id: str
    category: str
    severity: str
    clause_id: str
    clause_label: str
    explanation: str
    recommendation: str
    evidence: str


@dataclass
class RiskReport:
    contract_id: str
    title: str
    score: int
    level: str
    findings: list[RiskFinding]
    review_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
