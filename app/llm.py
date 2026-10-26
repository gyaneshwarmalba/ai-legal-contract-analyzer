"""Answer generation over retrieved clauses.

Generators receive only the retrieved clauses and must cite them as [n]. The
pipeline validates citations afterwards, so an answer that cites nothing (or
cites a clause it was not given) is marked ungrounded and routed to review.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from app.embeddings import tokenize
from app.models import RetrievedClause

logger = logging.getLogger(__name__)

INSUFFICIENT = "The provided contract clauses do not contain enough information to answer this question."

SYSTEM_PROMPT = (
    "You are a contract analysis assistant supporting a legal reviewer. "
    "Answer ONLY from the numbered contract clauses provided. "
    "Cite every statement with the clause number in square brackets, e.g. [1] or [2][3]. "
    f'If the clauses do not answer the question, reply exactly: "{INSUFFICIENT}" '
    "Do not give legal advice; describe what the contract says."
)


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[RetrievedClause]) -> str: ...


def format_context(contexts: list[RetrievedClause]) -> str:
    return "\n\n".join(f"[{i}] {rc.clause.citation_label}\n{rc.clause.text}" for i, rc in enumerate(contexts, start=1))


def parse_citations(text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", text)})


class ExtractiveGenerator:
    """Offline generator: selects the clause sentences that best overlap the question.

    It never invents text, which makes it a useful baseline for evaluating a
    hosted model against the same retrieval results.
    """

    name = "local-extractive"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: list[RetrievedClause]) -> str:
        q_tokens = set(tokenize(question))
        if not q_tokens or not contexts:
            return INSUFFICIENT
        scored: list[tuple[float, int, str]] = []
        for i, rc in enumerate(contexts, start=1):
            for sentence in re.split(r"(?<=[.;])\s+", rc.clause.text):
                s_tokens = set(tokenize(sentence))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / len(q_tokens)
                heading_bonus = 0.25 * len(q_tokens & set(tokenize(rc.clause.heading))) / len(q_tokens)
                scored.append((overlap + heading_bonus + 0.1 * rc.score, i, sentence.strip()))
        scored.sort(key=lambda t: -t[0])
        picked = [s for s in scored if s[0] > 0.15][: self.max_sentences]
        if not picked:
            return INSUFFICIENT
        return " ".join(f"{sentence} [{idx}]" for _, idx, sentence in picked)


class BedrockGenerator:
    """Claude on AWS Bedrock via the Converse API."""

    def __init__(self, model_id: str, region: str, max_tokens: int = 600) -> None:
        import boto3

        self._client = boto3.client("bedrock-runtime", region_name=region)
        self.name = model_id
        self.max_tokens = max_tokens

    def generate(self, question: str, contexts: list[RetrievedClause]) -> str:
        if not contexts:
            return INSUFFICIENT
        prompt = f"Contract clauses:\n\n{format_context(contexts)}\n\nQuestion: {question}"
        try:
            resp = self._client.converse(
                modelId=self.name,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
            )
        except Exception:
            logger.exception("bedrock converse request failed", extra={"model_id": self.name})
            raise
        return "".join(block.get("text", "") for block in resp["output"]["message"]["content"]).strip()
