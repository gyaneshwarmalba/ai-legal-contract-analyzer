"""ContractAnalyzer: wires ingestion, retrieval, generation, risk, and review together."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict
from pathlib import Path

from app.config import Settings
from app.embeddings import BedrockEmbedder, Embedder, HashingEmbedder
from app.ingest import contract_from_text, load_contract
from app.llm import INSUFFICIENT, BedrockGenerator, ExtractiveGenerator, Generator, parse_citations
from app.models import Answer, Citation, Contract, RiskReport
from app.review import ReviewItem, ReviewQueue
from app.risk import RiskPolicy
from app.sinks import LocalJsonSink, ResultSink, SalesforceSink
from app.vectorstore import FaissVectorStore, InMemoryVectorStore, VectorStore

logger = logging.getLogger(__name__)


def build_embedder(s: Settings) -> Embedder:
    if s.embeddings_provider == "bedrock":
        return BedrockEmbedder(s.bedrock_embed_model_id, s.aws_region)
    if s.embeddings_provider == "local":
        return HashingEmbedder()
    raise ValueError(f"unknown EMBEDDINGS_PROVIDER {s.embeddings_provider!r}")


def build_generator(s: Settings) -> Generator:
    if s.llm_provider == "bedrock":
        return BedrockGenerator(s.bedrock_model_id, s.aws_region)
    if s.llm_provider == "local":
        return ExtractiveGenerator()
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}")


def build_store(s: Settings, dim: int) -> VectorStore:
    if s.vector_store == "faiss":
        return FaissVectorStore(dim)
    if s.vector_store == "memory":
        return InMemoryVectorStore(dim)
    raise ValueError(f"unknown VECTOR_STORE {s.vector_store!r}")


def build_sink(s: Settings) -> ResultSink:
    if s.result_sink == "salesforce":
        return SalesforceSink()
    if s.result_sink == "local":
        return LocalJsonSink(s.state_dir / "published")
    raise ValueError(f"unknown RESULT_SINK {s.result_sink!r}")


class ContractAnalyzer:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
        store: VectorStore | None = None,
        sink: ResultSink | None = None,
        review_queue: ReviewQueue | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.embedder = embedder or build_embedder(self.settings)
        self.generator = generator or build_generator(self.settings)
        self.store = store or build_store(self.settings, self.embedder.dim)
        self.sink = sink or build_sink(self.settings)
        self.reviews = review_queue or ReviewQueue(self.settings.state_dir / "reviews.json")
        self.policy = RiskPolicy.load(self.settings.risk_rules_path)
        self.contracts: dict[str, Contract] = {}
        self._lock = threading.Lock()
        if self.settings.seed_sample_data:
            self.ingest_directory(self.settings.sample_contracts_dir)

    # ---- ingestion -------------------------------------------------------
    def add_contract(self, contract: Contract) -> Contract:
        if not contract.clauses:
            raise ValueError(f"contract {contract.contract_id!r} has no text to index")
        vectors = self.embedder.embed([f"{c.heading}\n{c.text}" for c in contract.clauses])
        with self._lock:
            if contract.contract_id in self.contracts:
                self.store.remove_contract(contract.contract_id)
            self.store.add(contract.clauses, vectors)
            self.contracts[contract.contract_id] = contract
        logger.info("contract indexed", extra=contract.summary())
        return contract

    def ingest_text(self, name: str, text: str) -> Contract:
        return self.add_contract(contract_from_text(name, text))

    def ingest_file(self, path: Path) -> Contract:
        return self.add_contract(load_contract(path))

    def ingest_directory(self, directory: Path) -> list[Contract]:
        loaded = []
        for path in sorted(directory.glob("*")):
            if path.suffix.lower() not in {".txt", ".md", ".docx"}:
                continue
            try:
                loaded.append(self.ingest_file(path))
            except (ValueError, RuntimeError, OSError):
                logger.exception("failed to ingest contract", extra={"path": str(path)})
        return loaded

    # ---- question answering ---------------------------------------------
    def ask(self, question: str, contract_id: str | None = None, k: int | None = None) -> Answer:
        if contract_id and contract_id not in self.contracts:
            raise KeyError(contract_id)
        query = self.embedder.embed([question])[0]
        hits = self.store.search(query, k or self.settings.top_k, contract_id=contract_id)
        contexts = [h for h in hits if h.score >= self.settings.min_relevance]
        text = self.generator.generate(question, contexts)

        cited = parse_citations(text)
        valid = [i for i in cited if 1 <= i <= len(contexts)]
        refused = text.strip() == INSUFFICIENT
        grounded = refused or (bool(valid) and len(valid) == len(cited))
        citations = [
            Citation(
                index=i,
                contract_id=contexts[i - 1].clause.contract_id,
                clause_id=contexts[i - 1].clause.clause_id,
                label=contexts[i - 1].clause.citation_label,
                score=round(contexts[i - 1].score, 4),
                excerpt=contexts[i - 1].clause.text[:400],
            )
            for i in valid
        ]
        answer = Answer(
            question=question,
            answer=text,
            citations=citations,
            grounded=grounded,
            model=self.generator.name,
            needs_review=not grounded,
        )
        if not grounded:
            item = self.reviews.submit(
                "answer",
                contract_id or "all",
                "Answer cites no clause or cites a clause that was not retrieved",
                {"question": question, "answer": text, "cited": cited, "retrieved": [h.clause.clause_id for h in contexts]},
            )
            answer.review_id = item.id
        logger.info(
            "question answered",
            extra={"contract_id": contract_id, "grounded": grounded, "citations": len(citations), "model": answer.model},
        )
        return answer

    # ---- risk ------------------------------------------------------------
    def assess(self, contract_id: str) -> RiskReport:
        contract = self.contracts.get(contract_id)
        if contract is None:
            raise KeyError(contract_id)
        report = self.policy.assess(contract)
        payload = {k: v for k, v in report.to_dict().items() if k != "review_id"}
        payload["policy_version"] = self.policy.version
        item = self.reviews.submit("risk", contract_id, f"{report.level} risk ({report.score}/100)", payload)
        report.review_id = item.id
        return report

    # ---- review ----------------------------------------------------------
    def decide(self, review_id: str, approve: bool, reviewer: str, note: str | None = None) -> tuple[ReviewItem, str | None]:
        item = self.reviews.decide(review_id, approve, reviewer, note)
        published_to = self.sink.publish(asdict(item)) if approve else None
        return item, published_to
