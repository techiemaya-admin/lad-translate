from lad_translate.chunker import ChunkerConfig, CommitReason, PhraseChunker
from tests.conftest import HYPOTHESIS_INTERVAL, growing, stream


def run(hyps, config=None) -> list:
    chunker = PhraseChunker(config)
    out = []
    for hyp in hyps:
        out.extend(chunker.feed(hyp))
    return out


def test_commits_at_a_clause_boundary():
    sentence = "Good morning everyone and welcome, today we will look at the results."
    chunks = run(growing(sentence))
    clause = [c for c in chunks if c.reason is CommitReason.CLAUSE]
    assert clause, "a comma past min_words should produce a clause commit"
    assert clause[0].text == "Good morning everyone and welcome,"


def test_reassembled_text_matches_the_source_exactly():
    sentence = "Good morning everyone and welcome, today we will look at the results."
    chunks = run(growing(sentence))
    assert " ".join(c.text for c in chunks) == sentence, "no words lost or duplicated"


def test_run_on_speech_is_forced_out_at_max_words():
    config = ChunkerConfig(max_words=8, min_words=4)
    sentence = " ".join(f"word{i}" for i in range(30))
    chunks = run(growing(sentence), config)
    forced = [c for c in chunks if c.reason is CommitReason.MAX_WORDS]
    assert forced, "speech with no boundaries must still be released"
    assert all(c.word_count <= 8 for c in chunks)


def test_stable_but_unbounded_text_is_released_on_timeout():
    config = ChunkerConfig(max_wait_s=1.0, min_words=4, max_words=25)
    # Backend has settled on five words and stopped producing anything new.
    stages = ["alpha beta gamma delta epsilon"] * 10
    chunks = run(stream(stages, final=False), config)
    assert chunks, "a stable prefix must not be held indefinitely"
    assert chunks[0].reason is CommitReason.STABILITY_TIMEOUT
    assert chunks[0].stability_lag >= config.max_wait_s


def test_silence_flushes_pending_text():
    config = ChunkerConfig(silence_commit_s=0.6)
    chunker = PhraseChunker(config)
    hyps = list(stream(["the speaker paused right", "the speaker paused right"], final=False))
    for hyp in hyps:
        chunker.feed(hyp)
    quiet = hyps[-1].t_wall + 1.0
    chunks = chunker.tick(quiet)
    assert chunks, "silence is a boundary the backend will not report"
    assert chunks[0].reason is CommitReason.SILENCE


def test_tick_before_the_silence_window_does_nothing():
    chunker = PhraseChunker(ChunkerConfig(silence_commit_s=0.6))
    hyps = list(stream(["one two three four", "one two three four"], final=False))
    for hyp in hyps:
        chunker.feed(hyp)
    assert chunker.tick(hyps[-1].t_wall + 0.1) == []


def test_revised_words_are_never_committed():
    # The classic failure: "do buy" becomes "Dubai" one hypothesis later.
    stages = [
        "we are going to do buy",
        "we are going to Dubai",
        "we are going to Dubai next month,",
        "we are going to Dubai next month, for the summit.",
    ]
    chunks = run(stream(stages))
    spoken = " ".join(c.text for c in chunks)
    assert "do buy" not in spoken, "a revised hypothesis reached the audience"
    assert "Dubai" in spoken


def test_final_hypothesis_drains_everything():
    chunks = run(growing("just a short line"))
    assert chunks, "a final must not leave text stuck in the buffer"
    assert chunks[-1].reason is CommitReason.FINAL
    assert " ".join(c.text for c in chunks) == "just a short line"


def test_chunk_ids_are_sequential_and_audio_spans_are_contiguous():
    sentence = "Good morning everyone and welcome, today we will look at the results."
    chunks = run(growing(sentence))
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    for previous, current in zip(chunks, chunks[1:]):
        assert current.t_audio_start == previous.t_audio_end


def test_stability_lag_is_recorded_for_every_chunk():
    chunks = run(growing("Good morning everyone and welcome, today we begin."))
    assert all(c.stability_lag >= 0.0 for c in chunks)
    # With a clean appending backend, agreement costs about one interval.
    clause = [c for c in chunks if c.reason is CommitReason.CLAUSE]
    assert all(c.stability_lag <= HYPOTHESIS_INTERVAL * 3 for c in clause)
