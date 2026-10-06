"""Text embedding behind a small interface, so tests can run without the model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

Vector = Sequence[float]


class Embedder(Protocol):
    """Turns texts into fixed-length vectors. Implementations must be deterministic."""

    @property
    def model_name(self) -> str:
        """Identifier recorded on the collection, to refuse mixing models."""
        ...

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed ``texts``, returning one vector per text in the same order."""
        ...


class MiniLmOnnxEmbedder:
    """``sentence-transformers/all-MiniLM-L6-v2`` run through ONNX Runtime on CPU.

    Same model and 384-dimension output as the Sentence Transformers package,
    but executed by Chroma's bundled ONNX build, so no torch install is needed
    (torch has no current Intel-Mac wheels). The model (~80 MB) downloads to
    ``~/.cache/chroma`` on first use. Input beyond 256 tokens is truncated,
    which is why chunks are kept small.
    """

    model_name = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        # Imported lazily: loading the ONNX session is slow and unit tests
        # that inject a fake embedder never need it.
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._function = ONNXMiniLM_L6_V2()

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed ``texts`` with the ONNX model.

        :param texts: Texts to embed.
        :type texts: Sequence[str]
        :returns: One 384-dimension vector per text.
        :rtype: list[Vector]
        """
        return [list(map(float, vector)) for vector in self._function(list(texts))]