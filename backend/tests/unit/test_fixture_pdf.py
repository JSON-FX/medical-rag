def test_generated_pdf_survives_parentheses_and_backslashes(tmp_path):
    """FDA label text is full of parens; an unbalanced one corrupts the PDF."""
    from pypdf import PdfReader

    from tests.fixtures.make_fixture_pdf import make_pdf

    tricky = r"Starting dose 500 mg (2.1) with meals \ see (5.1) and (eGFR below 30)"
    path = make_pdf(tmp_path / "tricky.pdf", [tricky])
    text = PdfReader(str(path)).pages[0].extract_text() or ""
    assert "2.1" in text
    assert "eGFR" in text
