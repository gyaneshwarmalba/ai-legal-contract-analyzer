import pytest

from app.llm import INSUFFICIENT
from app.models import RetrievedClause


@pytest.mark.parametrize(
    ("question", "contract_id", "expected_clause"),
    [
        ("Can the provider terminate for convenience?", "acme-master-services-agreement", "acme-master-services-agreement:5"),
        ("What is the cap on liability?", "globex-mutual-nda", "globex-mutual-nda:8"),
        ("How fast must the vendor notify us of a security breach?", "initech-saas-subscription", "initech-saas-subscription:6"),
        ("Does the agreement automatically renew?", "acme-master-services-agreement", "acme-master-services-agreement:3"),
        ("What is the uptime service level commitment?", None, "initech-saas-subscription:2"),
        ("How long do confidentiality obligations last?", "globex-mutual-nda", "globex-mutual-nda:5"),
    ],
)
def test_retrieval_finds_the_right_clause(analyzer, question, contract_id, expected_clause):
    answer = analyzer.ask(question, contract_id)
    assert answer.grounded
    assert answer.citations, answer.answer
    assert answer.citations[0].clause_id == expected_clause


def test_answers_cite_only_retrieved_clauses(analyzer):
    answer = analyzer.ask("When are invoices due?", "initech-saas-subscription")
    assert "[1]" in answer.answer
    assert "forty-five (45) days" in answer.answer


def test_unanswerable_question_refuses_without_review(analyzer):
    answer = analyzer.ask("zebra quantum photosynthesis", "globex-mutual-nda")
    assert answer.answer == INSUFFICIENT
    assert answer.grounded and not answer.needs_review


class HallucinatingGenerator:
    name = "bad-model"

    def generate(self, question, contexts: list[RetrievedClause]) -> str:
        return "The contract allows unlimited refunds [9]."


def test_invalid_citation_is_flagged_and_sent_to_review(settings):
    from app.pipeline import ContractAnalyzer

    analyzer = ContractAnalyzer(settings, generator=HallucinatingGenerator())
    answer = analyzer.ask("Are refunds allowed?")
    assert not answer.grounded
    assert answer.needs_review and answer.review_id
    assert analyzer.reviews.get(answer.review_id).kind == "answer"


def test_reingest_replaces_previous_version(analyzer):
    old_count = len(analyzer.contracts["globex-mutual-nda"].clauses)
    before = len(analyzer.store)
    analyzer.ingest_text("globex mutual nda", "NDA\n1. TERM\nOne year only.")
    assert len(analyzer.store) == before - old_count + 1
    hits = analyzer.ask("What is the term?", "globex-mutual-nda").citations
    assert [c.clause_id for c in hits] == ["globex-mutual-nda:1"]
