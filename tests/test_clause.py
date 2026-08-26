import pytest

from lad_translate.chunker.clause import BoundaryStrength, find_boundary, token_boundary_strength


@pytest.mark.parametrize(
    "token,expected",
    [
        ("results.", BoundaryStrength.STRONG),
        ("really?", BoundaryStrength.STRONG),
        ("wonderful!", BoundaryStrength.STRONG),
        ("welcome,", BoundaryStrength.MEDIUM),
        ("following:", BoundaryStrength.MEDIUM),
        ("however;", BoundaryStrength.MEDIUM),
        ("conference", BoundaryStrength.NONE),
    ],
)
def test_boundary_strength(token, expected):
    assert token_boundary_strength(token) is expected


@pytest.mark.parametrize("token", ["Mr.", "Dr.", "etc.", "Ltd.", "J.", "3.5", "2."])
def test_abbreviations_and_numbers_are_not_sentence_ends(token):
    assert token_boundary_strength(token) is not BoundaryStrength.STRONG


def test_arabic_and_devanagari_punctuation_is_recognised():
    assert token_boundary_strength("مرحبا،") is BoundaryStrength.MEDIUM
    assert token_boundary_strength("नमस्ते।") is BoundaryStrength.STRONG


def test_latest_boundary_wins_so_chunks_are_as_long_as_possible():
    tokens = ["first", "clause,", "second", "clause,", "third", "part"]
    end, strength = find_boundary(tokens, min_words=3)
    assert strength is BoundaryStrength.MEDIUM
    assert tokens[:end] == ["first", "clause,", "second", "clause,"]


def test_boundary_below_min_words_is_rejected():
    tokens = ["yes,", "absolutely"]
    assert find_boundary(tokens, min_words=4) == (0, BoundaryStrength.NONE)


def test_weak_boundary_sits_before_the_conjunction():
    tokens = ["we", "looked", "at", "the", "numbers", "and", "they", "were", "encouraging"]
    assert find_boundary(tokens, min_words=4, allow_weak=False)[0] == 0
    end, strength = find_boundary(tokens, min_words=4, allow_weak=True)
    assert strength is BoundaryStrength.WEAK
    assert tokens[end] == "and", "the conjunction must open the next chunk"
