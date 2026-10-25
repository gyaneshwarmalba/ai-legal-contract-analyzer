import pytest
from fastapi.testclient import TestClient

from app.main import app, get_analyzer


@pytest.fixture()
def client(analyzer):
    app.dependency_overrides[get_analyzer] = lambda: analyzer
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["contracts"] == 3


def test_ask_and_review_flow(client):
    ans = client.post("/ask", json={"question": "Who can terminate and on what notice?", "contract_id": "globex-mutual-nda"})
    assert ans.status_code == 200 and ans.json()["citations"]

    risk = client.get("/contracts/acme-master-services-agreement/risk").json()
    assert risk["level"] == "high"
    pending = client.get("/reviews", params={"status": "pending"}).json()
    assert any(r["id"] == risk["review_id"] for r in pending)

    decision = client.post(f"/reviews/{risk['review_id']}/decision", json={"approve": True, "reviewer": "alice"})
    assert decision.status_code == 200 and decision.json()["published_to"]
    again = client.post(f"/reviews/{risk['review_id']}/decision", json={"approve": True, "reviewer": "alice"})
    assert again.status_code == 409


def test_add_contract_and_errors(client):
    created = client.post("/contracts", json={"name": "Tiny Lease", "text": "LEASE\n1. RENT\nRent is due monthly on the first."})
    assert created.status_code == 201 and created.json()["contract_id"] == "tiny-lease"
    assert client.get("/contracts/tiny-lease/clauses").json()[0]["heading"] == "Rent"
    assert client.get("/contracts/nope/risk").status_code == 404
    assert client.post("/ask", json={"question": "anything?", "contract_id": "nope"}).status_code == 404
    assert client.post("/contracts", json={"name": "x", "text": "short"}).status_code == 422
