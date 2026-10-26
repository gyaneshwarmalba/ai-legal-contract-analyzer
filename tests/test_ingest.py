from app.ingest import chunk_contract, slugify


def test_splits_on_numbered_headings_and_keeps_subsections_together():
    text = """SAMPLE AGREEMENT

Intro paragraph.

1. DEFINITIONS
Terms mean things.

2. FEES AND PAYMENT
2.1 Pay on time.
2.2 Late fees apply.
"""
    title, clauses = chunk_contract("sample", text)
    assert title == "SAMPLE AGREEMENT"
    assert [c.number for c in clauses] == ["0", "1", "2"]
    assert clauses[2].heading == "Fees And Payment"
    assert "2.1 Pay on time." in clauses[2].text and "2.2 Late fees apply." in clauses[2].text
    assert clauses[2].clause_id == "sample:2"


def test_section_and_article_headings():
    text = "Doc\nSection 3 Term\nLasts a year.\nARTICLE IV Law\nOregon."
    _, clauses = chunk_contract("x", text)
    assert [(c.number, c.heading) for c in clauses] == [("3", "Term"), ("IV", "Law")]


def test_long_clause_is_split_into_parts():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    _, clauses = chunk_contract("long", f"Doc\n1. SCOPE\n{body}")
    assert len(clauses) > 1
    assert all(c.number == "1" for c in clauses)
    assert clauses[0].clause_id == "long:1-1"


def test_stemmer_is_consistent_across_word_forms():
    from app.embeddings import stem

    assert stem("obligations") == stem("obligation")
    assert stem("terminate") == stem("termination") == stem("terminated")
    assert stem("clauses") == stem("clause")
    assert stem("liabilities") == stem("liability")


def test_slugify():
    assert slugify("Acme MSA (2025).docx") == "acme-msa-2025-docx"
