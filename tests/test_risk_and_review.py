import json

import pytest


def rule_ids(report):
    return {f.rule_id for f in report.findings}


def test_one_sided_msa_scores_high(analyzer):
    report = analyzer.assess("acme-master-services-agreement")
    assert report.level == "high"
    assert {"LIAB-001", "INDEM-001", "TERM-001", "RENEW-001", "DATA-001", "SUB-001"} <= rule_ids(report)
    liab = next(f for f in report.findings if f.rule_id == "LIAB-001")
    assert liab.clause_id == "acme-master-services-agreement:8"
    assert "liable for all" in liab.evidence


def test_balanced_contracts_score_low(analyzer):
    for cid in ("globex-mutual-nda", "initech-saas-subscription"):
        report = analyzer.assess(cid)
        assert report.level == "low", (cid, rule_ids(report))
        assert not {"LIAB-001", "INDEM-001", "DATA-001", "TERM-001"} & rule_ids(report)


def test_findings_sorted_by_severity(analyzer):
    severities = [f.severity for f in analyzer.assess("acme-master-services-agreement").findings]
    order = {"high": 0, "medium": 1, "low": 2}
    assert severities == sorted(severities, key=order.get)


def test_assessment_is_queued_once_and_published_only_after_approval(analyzer, settings):
    first = analyzer.assess("acme-master-services-agreement")
    second = analyzer.assess("acme-master-services-agreement")
    assert first.review_id == second.review_id
    published_dir = settings.state_dir / "published"
    assert not published_dir.exists()

    item, dest = analyzer.decide(first.review_id, approve=True, reviewer="alice", note="escalate liability")
    assert item.status == "approved"
    record = json.loads(open(dest).read())
    assert record["payload"]["score"] == first.score and record["reviewer"] == "alice"

    with pytest.raises(ValueError):
        analyzer.decide(first.review_id, approve=False, reviewer="bob")


def test_rejection_does_not_publish(analyzer, settings):
    report = analyzer.assess("globex-mutual-nda")
    _, dest = analyzer.decide(report.review_id, approve=False, reviewer="alice")
    assert dest is None
    assert not (settings.state_dir / "published").exists()


def test_review_queue_survives_restart(analyzer, settings):
    from app.review import ReviewQueue

    report = analyzer.assess("initech-saas-subscription")
    reloaded = ReviewQueue(settings.state_dir / "reviews.json")
    assert reloaded.get(report.review_id).status == "pending"
