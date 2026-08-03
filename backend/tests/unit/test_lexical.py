import pytest

from rag.lexical import build_fts_query


def test_simple_question_becomes_or_of_quoted_terms():
    assert build_fts_query("metformin dose") == '"metformin" OR "dose"'


@pytest.mark.parametrize(
    "question",
    [
        "What's the max dose?",
        'He said "take two"',
        "dose*",
        "metformin - adult",
        "dose NEAR adult",
        "metformin AND atenolol",
        "a OR b NOT c",
        "50% w/v (10:1)",
        "^caret $dollar",
    ],
)
def test_fts_syntax_characters_never_survive_sanitising(question):
    """Raw questions raise `fts5: syntax error`; this is the guard."""
    result = build_fts_query(question)
    unquoted = result.replace('"', "").replace(" OR ", " ")
    assert all(ch.isalnum() or ch.isspace() for ch in unquoted), result


def test_reserved_words_are_quoted_so_they_are_literals():
    result = build_fts_query("dose NEAR adult")
    assert '"near"' in result
    assert " NEAR " not in result


def test_single_character_terms_are_dropped_as_noise():
    assert build_fts_query("a b metformin") == '"metformin"'


def test_terms_are_deduplicated_preserving_first_occurrence():
    assert build_fts_query("dose dose metformin dose") == '"dose" OR "metformin"'


def test_empty_or_punctuation_only_question_yields_empty_string():
    assert build_fts_query("") == ""
    assert build_fts_query("???  !!!") == ""
    assert build_fts_query("a") == ""


def test_unicode_terms_are_preserved():
    assert '"naïve"' in build_fts_query("naïve dosing")


def test_numbers_are_kept_because_dosages_matter():
    assert '"500mg"' in build_fts_query("is it 500mg")
