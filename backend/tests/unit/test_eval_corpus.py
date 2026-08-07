import pathlib

from pypdf import PdfReader

from evals.corpus import assemble_text, corpus_text, build_pdf


SECTIONS = {
    "dosage_and_administration": "Adult dose is 500 mg (2.1) twice daily.",
    "contraindications": "Contraindicated below eGFR 30 mL/min/1.73 m 2 .",
}


def test_corpus_text_includes_the_title_and_every_section():
    text = corpus_text("Metformin", SECTIONS)
    assert "Metformin" in text
    assert "500 mg" in text
    assert "eGFR" in text


def test_assemble_text_is_stable_across_calls():
    assert assemble_text(SECTIONS) == assemble_text(SECTIONS)


def test_build_pdf_round_trips_text_containing_parentheses(tmp_path):
    """FDA text is dense with parens; an unbalanced one corrupts the PDF."""
    path = build_pdf(tmp_path / "d.pdf", "Metformin", SECTIONS)
    extracted = " ".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    assert "500 mg" in extracted
    assert "eGFR" in extracted


def test_build_pdf_loses_no_words(tmp_path):
    """Pagination splits on whitespace; nothing may be dropped."""
    path = build_pdf(tmp_path / "d.pdf", "Metformin", SECTIONS)
    extracted = " ".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    for word in corpus_text("Metformin", SECTIONS).split():
        assert word in extracted, f"lost {word!r}"
