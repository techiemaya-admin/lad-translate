from lad_translate.chunker import StabilityTracker
from tests.conftest import stream


def test_no_prefix_until_agreement_reached():
    tracker = StabilityTracker(agreement_n=2)
    hyps = list(stream(["the quick"], final=False))
    tracker.add(hyps[0])
    assert tracker.stable_prefix() is None, "one hypothesis cannot agree with itself"


def test_agreement_two_yields_common_prefix():
    tracker = StabilityTracker(agreement_n=2)
    for hyp in stream(["the quick brown", "the quick brown fox"], final=False):
        tracker.add(hyp)
    prefix = tracker.stable_prefix()
    assert prefix is not None
    assert prefix.tokens == ["the", "quick", "brown"]


def test_revision_withholds_the_revised_words():
    tracker = StabilityTracker(agreement_n=2)
    # "do buy" is revised to "Dubai". Only the agreed head may be released.
    for hyp in stream(["we are going to do buy", "we are going to Dubai"], final=False):
        tracker.add(hyp)
    prefix = tracker.stable_prefix()
    assert prefix is not None
    assert prefix.tokens == ["we", "are", "going", "to"]


def test_punctuation_churn_does_not_reset_stability():
    tracker = StabilityTracker(agreement_n=2)
    for hyp in stream(["the quick brown fox", "The quick, brown fox."], final=False):
        tracker.add(hyp)
    prefix = tracker.stable_prefix()
    assert prefix is not None
    assert len(prefix.tokens) == 4
    # Surface text comes from the newest hypothesis, so punctuation survives.
    assert prefix.text == "The quick, brown fox."


def test_contradiction_after_commit_is_flagged():
    tracker = StabilityTracker(agreement_n=2)
    for hyp in stream(["alpha beta", "alpha beta"], final=False):
        tracker.add(hyp)
    tracker.commit(2)
    for hyp in stream(["alpha gamma delta", "alpha gamma delta"], final=False):
        tracker.add(hyp)
    assert tracker.saw_contradiction, "committed text was contradicted and must be reported"
