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


def test_decimal_dosages_are_not_conflated():
    """0.5mg and 5mg must not produce the same query — in a dosing context that
    is the difference between a pediatric and an adult dose."""
    assert build_fts_query("0.5mg") != build_fts_query("5mg")
    assert '"0.5mg"' in build_fts_query("is the dose 0.5mg")
    assert '"2.5mg"' in build_fts_query("2.5mg twice daily")


def test_whole_number_dosages_still_tokenise():
    assert '"500mg"' in build_fts_query("is it 500mg")
    assert '"50"' in build_fts_query("atenolol 50 mg")


def test_stopwords_are_dropped():
    """Function words make every question match every chunk once OR-joined."""
    result = build_fts_query("what is the dose of metformin for an adult")
    for word in ("what", "is", "the", "of", "for", "an"):
        assert f'"{word}"' not in result, f"stopword {word!r} survived"
    assert '"dose"' in result and '"metformin"' in result and '"adult"' in result


def test_a_question_of_only_stopwords_yields_empty():
    assert build_fts_query("what is the of for an") == ""
