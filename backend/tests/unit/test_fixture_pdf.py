def test_generated_pdf_survives_parentheses_and_backslashes(tmp_path):
    """FDA label text is full of parens; an unbalanced one corrupts the PDF.

    The input has to be genuinely unbalanced. PDF string literals allow BALANCED
    parens through unescaped, so a fixture like "(2.1) and (5.1)" parses fine
    with `_escape` replaced by the identity function — the earlier version of
    this test passed against an `_escape` that did nothing. The lone `(` below
    swallows the closing delimiter and the reader runs off the end of the
    stream, so only real escaping keeps the file readable.

    The backslash is asserted for the same reason: unescaped, PDF treats `\\ `
    as an undefined escape and silently drops it, which no assertion on the
    surrounding words would notice.
    """
    from pypdf import PdfReader

    from tests.fixtures.make_fixture_pdf import make_pdf

    tricky = r"Reduce dose (see 5.1 and eGFR below 30 \ per label"
    path = make_pdf(tmp_path / "tricky.pdf", [tricky])
    text = PdfReader(str(path)).pages[0].extract_text() or ""
    assert "(see 5.1" in text
    assert "eGFR below 30" in text
    assert "\\" in text
