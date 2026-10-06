import random

import pytest

from app.chunking import TextChunk, chunk_text


def words(count: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "Normandy", "1066"]
    return " ".join(rng.choice(vocab) for _ in range(count))


def assert_well_formed(text: str, chunks: list[TextChunk], size: int) -> None:
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert 0 < len(chunk.text) <= size
        assert chunk.text == chunk.text.strip()
    # Every non-whitespace character is inside at least one chunk.
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start, chunk.end))
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())


def test_short_text_is_one_chunk_with_outer_whitespace_trimmed() -> None:
    chunks = chunk_text("  The Normans were in Normandy.\n", chunk_size=1000, chunk_overlap=200)

    assert chunks == [TextChunk(index=0, start=2, end=31, text="The Normans were in Normandy.")]


def test_long_text_splits_within_size_and_covers_everything() -> None:
    text = words(600)

    chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)

    assert len(chunks) > 1
    assert_well_formed(text, chunks, 300)


def test_chunks_never_cut_a_word() -> None:
    text = words(600)
    vocab_words = set(text.split())

    for chunk in chunk_text(text, chunk_size=250, chunk_overlap=50):
        assert set(chunk.text.split()) <= vocab_words
        assert chunk.start == 0 or text[chunk.start - 1].isspace()
        assert chunk.end == len(text) or text[chunk.end].isspace()


def test_neighbouring_chunks_overlap() -> None:
    text = words(600)

    chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start < previous.end
        assert current.start > previous.start


def test_zero_overlap_produces_disjoint_chunks() -> None:
    text = words(600)

    chunks = chunk_text(text, chunk_size=300, chunk_overlap=0)

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start >= previous.end
    assert_well_formed(text, chunks, 300)


def test_word_longer_than_chunk_size_is_hard_cut() -> None:
    text = "short " + "x" * 25 + " tail"

    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)

    assert_well_formed(text, chunks, 10)


def test_chunking_is_deterministic() -> None:
    text = words(800)

    assert chunk_text(text, 400, 80) == chunk_text(text, 400, 80)


@pytest.mark.parametrize(
    ("text", "size", "overlap", "message"),
    [
        ("   \n\t ", 100, 10, "blank"),
        ("text", 0, 0, "chunk_size"),
        ("text", 100, 100, "chunk_overlap"),
        ("text", 100, -1, "chunk_overlap"),
    ],
)
def test_invalid_input_is_rejected(text: str, size: int, overlap: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text(text, size, overlap)