"""Reviewer UI: `streamlit run ui/streamlit_app.py`."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import configure_logging  # noqa: E402
from app.pipeline import ContractAnalyzer  # noqa: E402

configure_logging()
st.set_page_config(page_title="AI Legal Contract Analyzer", layout="wide")


@st.cache_resource
def analyzer() -> ContractAnalyzer:
    return ContractAnalyzer()


a = analyzer()
st.title("AI Legal Contract Analyzer")
st.caption(f"Generator: `{a.generator.name}` · Embeddings: `{a.embedder.name}` · Policy: `{a.policy.version}`")

contracts = {c.contract_id: c for c in a.contracts.values()}
ask_tab, risk_tab, review_tab, upload_tab = st.tabs(["Ask", "Risk report", "Review queue", "Add contract"])

with ask_tab:
    scope = st.selectbox("Contract", ["All contracts", *contracts], key="ask_scope")
    question = st.text_input("Question", "Can the provider terminate the agreement for convenience?")
    if st.button("Ask", type="primary") and question.strip():
        ans = a.ask(question, None if scope == "All contracts" else scope)
        (st.success if ans.grounded else st.warning)(ans.answer)
        if ans.needs_review:
            st.info(f"Answer was not grounded in retrieved clauses and was sent to review ({ans.review_id}).")
        for c in ans.citations:
            with st.expander(f"[{c.index}] {c.label} (similarity {c.score:.2f})"):
                st.write(c.excerpt)

with risk_tab:
    if contracts:
        cid = st.selectbox("Contract", list(contracts), key="risk_scope")
        report = a.assess(cid)
        cols = st.columns(3)
        cols[0].metric("Risk score", f"{report.score}/100")
        cols[1].metric("Level", report.level.upper())
        cols[2].metric("Findings", len(report.findings))
        for f in report.findings:
            with st.expander(f"{f.severity.upper()} · {f.category} · {f.clause_label}"):
                st.markdown(f"**Why:** {f.explanation}")
                st.markdown(f"**Evidence:** _{f.evidence}_")
                st.markdown(f"**Recommendation:** {f.recommendation}")
        st.caption(f"Submitted to review queue as `{report.review_id}`")

with review_tab:
    pending = a.reviews.list("pending")
    st.write(f"{len(pending)} pending item(s)")
    reviewer = st.text_input("Reviewer name", "legal.reviewer")
    for item in pending:
        with st.expander(f"{item.kind} · {item.contract_id} · {item.reason}"):
            st.json(item.payload, expanded=False)
            note = st.text_input("Note", key=f"note-{item.id}")
            c1, c2 = st.columns(2)
            if c1.button("Approve", key=f"ok-{item.id}"):
                _, dest = a.decide(item.id, True, reviewer, note or None)
                st.success(f"Approved and published to {dest}")
                st.rerun()
            if c2.button("Reject", key=f"no-{item.id}"):
                a.decide(item.id, False, reviewer, note or None)
                st.rerun()

with upload_tab:
    name = st.text_input("Contract name", "my-contract")
    text = st.text_area("Contract text", height=300)
    if st.button("Index contract") and text.strip():
        try:
            c = a.ingest_text(name, text)
            st.success(f"Indexed {c.contract_id} with {len(c.clauses)} clauses")
        except ValueError as exc:
            st.error(str(exc))
