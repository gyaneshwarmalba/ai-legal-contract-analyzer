"""Rule-based clause risk scoring.

Rules live in versioned JSON (policies/risk_rules.json) owned by the legal
team, not in code. Each finding records the rule, the clause, and the exact
matching text so a reviewer can see why it fired.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.models import Contract, RiskFinding, RiskReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskRule:
    id: str
    category: str
    severity: str
    applies_to: str
    match: tuple[re.Pattern[str], ...]
    unless: tuple[re.Pattern[str], ...]
    explanation: str
    recommendation: str


class RiskPolicy:
    def __init__(self, rules: list[RiskRule], weights: dict[str, int], version: str) -> None:
        self.rules = rules
        self.weights = weights
        self.version = version

    @classmethod
    def load(cls, path: Path) -> RiskPolicy:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rules = []
        for r in raw["rules"]:
            if r["severity"] not in raw["severity_weights"]:
                raise ValueError(f"rule {r['id']} has unknown severity {r['severity']!r}")
            rules.append(
                RiskRule(
                    id=r["id"],
                    category=r["category"],
                    severity=r["severity"],
                    applies_to=r.get("applies_to", "").lower(),
                    match=tuple(re.compile(p, re.IGNORECASE) for p in r["match"]),
                    unless=tuple(re.compile(p, re.IGNORECASE) for p in r.get("unless", [])),
                    explanation=r["explanation"],
                    recommendation=r["recommendation"],
                )
            )
        logger.info("risk policy loaded", extra={"version": raw.get("version"), "rule_count": len(rules)})
        return cls(rules, raw["severity_weights"], raw.get("version", "unversioned"))

    def assess(self, contract: Contract) -> RiskReport:
        findings: list[RiskFinding] = []
        for clause in contract.clauses:
            haystack = f"{clause.heading}\n{clause.text}"
            for rule in self.rules:
                if rule.applies_to and rule.applies_to not in haystack.lower():
                    continue
                hit = next((m for p in rule.match if (m := p.search(haystack))), None)
                if not hit or any(p.search(haystack) for p in rule.unless):
                    continue
                findings.append(
                    RiskFinding(
                        rule_id=rule.id,
                        category=rule.category,
                        severity=rule.severity,
                        clause_id=clause.clause_id,
                        clause_label=clause.citation_label,
                        explanation=rule.explanation,
                        recommendation=rule.recommendation,
                        evidence=_evidence(haystack, hit.start(), hit.end()),
                    )
                )
        score = min(100, sum(self.weights[f.severity] for f in findings))
        return RiskReport(
            contract_id=contract.contract_id,
            title=contract.title,
            score=score,
            level=risk_level(score),
            findings=sorted(findings, key=lambda f: -self.weights[f.severity]),
        )


def risk_level(score: int) -> str:
    if score >= 60:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def _evidence(text: str, start: int, end: int, window: int = 90) -> str:
    lo, hi = max(0, start - window), min(len(text), end + window)
    snippet = " ".join(text[lo:hi].split())
    return ("…" if lo > 0 else "") + snippet + ("…" if hi < len(text) else "")
