"""FastAPI service: `uvicorn app.main:app --reload`, docs at http://localhost:8000/docs."""

from __future__ import annotations

import logging
from dataclasses import asdict
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.logging_config import configure_logging
from app.pipeline import ContractAnalyzer

configure_logging()
logger = logging.getLogger(__name__)


class ContractIn(BaseModel):
    name: str = Field(min_length=1, max_length=200, examples=["vendor-msa-2025"])
    text: str = Field(min_length=20, max_length=500_000)


class QuestionIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000, examples=["Can the provider terminate for convenience?"])
    contract_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)


class DecisionIn(BaseModel):
    approve: bool
    reviewer: str = Field(min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)


@lru_cache(maxsize=1)
def get_analyzer() -> ContractAnalyzer:
    return ContractAnalyzer()


app = FastAPI(title="AI Legal Contract Analyzer", version=__version__)


@app.get("/health")
def health(analyzer: ContractAnalyzer = Depends(get_analyzer)) -> dict:
    return {
        "status": "ok",
        "contracts": len(analyzer.contracts),
        "indexed_clauses": len(analyzer.store),
        "embedder": analyzer.embedder.name,
        "generator": analyzer.generator.name,
        "sink": analyzer.sink.name,
        "policy_version": analyzer.policy.version,
    }


@app.get("/contracts")
def list_contracts(analyzer: ContractAnalyzer = Depends(get_analyzer)) -> list[dict]:
    return [c.summary() for c in analyzer.contracts.values()]


@app.post("/contracts", status_code=201)
def add_contract(body: ContractIn, analyzer: ContractAnalyzer = Depends(get_analyzer)) -> dict:
    try:
        return analyzer.ingest_text(body.name, body.text).summary()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/contracts/{contract_id}/clauses")
def list_clauses(contract_id: str, analyzer: ContractAnalyzer = Depends(get_analyzer)) -> list[dict]:
    contract = analyzer.contracts.get(contract_id)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"unknown contract {contract_id}")
    return [asdict(c) for c in contract.clauses]


@app.get("/contracts/{contract_id}/risk")
def assess(contract_id: str, analyzer: ContractAnalyzer = Depends(get_analyzer)) -> dict:
    try:
        return analyzer.assess(contract_id).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown contract {contract_id}") from exc


@app.post("/ask")
def ask(body: QuestionIn, analyzer: ContractAnalyzer = Depends(get_analyzer)) -> dict:
    try:
        return analyzer.ask(body.question, body.contract_id, body.k).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown contract {body.contract_id}") from exc


@app.get("/reviews")
def list_reviews(status: str | None = None, analyzer: ContractAnalyzer = Depends(get_analyzer)) -> list[dict]:
    return [asdict(i) for i in analyzer.reviews.list(status)]


@app.post("/reviews/{review_id}/decision")
def decide(review_id: str, body: DecisionIn, analyzer: ContractAnalyzer = Depends(get_analyzer)) -> dict:
    try:
        item, published_to = analyzer.decide(review_id, body.approve, body.reviewer, body.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown review {review_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"review": asdict(item), "published_to": published_to}
