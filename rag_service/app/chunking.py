"""Split document text into overlapping, word-aligned chunks.

Pure and deterministic: the same text and parameters always give the same
chunks, which is what makes re-ingestion predictable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    """One chunk of a document.

    :ivar index: Zero-based position of the chunk within its document.
    :ivar start: Offset of the chunk's first character in the document text.
    :ivar end: Offset one past the chunk's last character.
    :ivar text: ``document_text[start:end]``, never empty and never starting
        or ending in whitespace.
    """

    index: int
    start: int
    end: int
    text: str


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Text that fits is returned as a single chunk. Longer text is cut at the
    last whitespace before the size limit, so words stay whole; a single word
    longer than ``chunk_size`` is the only thing ever cut mid-word. Each chunk
    after the first starts roughly ``chunk_overlap`` characters before the
    previous one ended, moved forward to the next word start. Only whitespace
    is ever left out, so every word of ``text`` lands in at least one chunk.

    :param text: Document text. Leading and trailing whitespace is ignored.
    :type text: str
    :param chunk_size: Maximum chunk length in characters.
    :type chunk_size: int
    :param chunk_overlap: Target number of characters shared by neighbouring
        chunks.
    :type chunk_overlap: int
    :returns: Chunks in document order, indexed from zero.
    :rtype: list[TextChunk]
    :raises ValueError: If ``text`` is blank, ``chunk_size`` is not positive,
        or ``chunk_overlap`` is negative or not smaller than ``chunk_size``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap} for size {chunk_size}"
        )

    if not text.strip():
        raise ValueError("text is blank")
    start = len(text) - len(text.lstrip())
    stop = len(text.rstrip())

    chunks: list[TextChunk] = []
    while True:
        hard_end = min(start + chunk_size, stop)
        end = hard_end
        if hard_end < stop:
            # Break at the last whitespace that keeps the chunk within size.
            boundary = next(
                (i for i in range(hard_end, start, -1) if text[i].isspace()),
                None,
            )
            if boundary is not None:
                end = boundary
        while text[end - 1].isspace():
            end -= 1
        chunks.append(TextChunk(index=len(chunks), start=start, end=end, text=text[start:end]))
        if hard_end >= stop:
            return chunks

        # Step back by the overlap, but always make progress.
        next_start = max(end - chunk_overlap, start + 1)
        # If that lands mid-word, move forward to the next word start.
        if not text[next_start - 1].isspace():
            while next_start < end and not text[next_start].isspace():
                next_start += 1
        while next_start < stop and text[next_start].isspace():
            next_start += 1
        start = next_start
