"""Contract loading and clause-aware chunking.

Contracts are split on their own numbered headings ("1. DEFINITIONS",
"Section 4 Fees", "ARTICLE III TERM") rather than on fixed token windows, so
each chunk is a clause a lawyer would recognise and every citation points to a
real section number.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.models import Clause, Contract

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d{1,2})\.\s+(?P<title>[A-Z][A-Z0-9 ,&'/()\-]{2,80})"
    r"|(?:Section|SECTION)\s+(?P<snum>\d{1,2}(?:\.\d+)?)[.:]?\s+(?P<stitle>.{2,80})"
    r"|(?:Article|ARTICLE)\s+(?P<anum>[IVXLC]+|\d{1,2})[.:]?\s+(?P<atitle>.{2,80})"
    r")\s*$"
)
MAX_CLAUSE_CHARS = 1500


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "contract"


def _match_heading(line: str) -> tuple[str, str] | None:
    m = HEADING_RE.match(line)
    if not m:
        return None
    if m.group("num"):
        return m.group("num"), m.group("title").strip().title()
    if m.group("snum"):
        return m.group("snum"), m.group("stitle").strip()
    return m.group("anum"), m.group("atitle").strip()


def _split_long(text: str, limit: int = MAX_CLAUSE_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n|\n(?=\d+\.\d+\s)", text):
        if current and len(current) + len(para) > limit:
            parts.append(current.strip())
            current = ""
        current += para + "\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_contract(contract_id: str, text: str) -> tuple[str, list[Clause]]:
    """Return (title, clauses). The first non-empty line is treated as the title."""
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    title = next((ln.strip() for ln in lines if ln.strip()), contract_id)

    sections: list[tuple[str, str, list[str]]] = [("0", "Preamble", [])]
    for line in lines:
        heading = _match_heading(line)
        body = sections[-1][2]
        if heading:
            sections.append((heading[0], heading[1], []))
        elif not line.strip():
            if body and body[-1]:
                body.append("")
        elif line.strip() != title:
            body.append(line.strip())

    clauses: list[Clause] = []
    for number, heading, body in sections:
        body_text = "\n".join(body).strip()
        if not body_text:
            continue
        parts = _split_long(body_text)
        for i, part in enumerate(parts, start=1):
            suffix = f"-{i}" if len(parts) > 1 else ""
            clauses.append(
                Clause(
                    contract_id=contract_id,
                    clause_id=f"{contract_id}:{number}{suffix}",
                    number=number,
                    heading=heading,
                    text=part,
                )
            )

    if len(sections) == 1:
        logger.warning("no clause headings detected; contract indexed as a single chunk", extra={"contract_id": contract_id})
    return title, clauses


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        try:
            import docx
        except ImportError as exc:
            raise RuntimeError("Reading .docx files requires python-docx (pip install python-docx)") from exc
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def load_contract(path: Path) -> Contract:
    contract_id = slugify(path.stem)
    title, clauses = chunk_contract(contract_id, read_document(path))
    return Contract(contract_id=contract_id, title=title, source=str(path.name), clauses=clauses)


def contract_from_text(name: str, text: str) -> Contract:
    contract_id = slugify(name)
    title, clauses = chunk_contract(contract_id, text)
    return Contract(contract_id=contract_id, title=title, source=name, clauses=clauses)
