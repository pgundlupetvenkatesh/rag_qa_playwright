#!/usr/bin/env python3
"""Load the SQuAD 2.0 validation passages into a running rag_service.

Run from ``rag_service/`` with the service up::

    python scripts/ingest_squad.py [--url http://127.0.0.1:8000] [--reset]

Only the ``title`` and ``context`` columns are read; questions, answers and
answerability never leave the dataset. The script talks to the service only
through ``POST /corpus/documents`` (and ``DELETE /corpus`` for ``--reset``), the
same API any other client uses.

The mapping follows the repo's ``RagTestCase`` contract
(``src/models/ragTestCase.ts``), which the framework side reads:

- document ``text`` is a passage, i.e. what a case carries in ``context[0]``;
- document ``id`` is :func:`passage_document_id` of that text, so a test can
  derive the expected id from a case's ``context`` without the service ever
  seeing the golden file;
- ``metadata.title`` matches ``RagTestCase.metadata.title``, and
  ``metadata.source`` matches ``RagTestCase.source`` (``squad_v2``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

DATASET_NAME = "rajpurkar/squad_v2"
SPLIT = "validation"
SOURCE = "squad_v2"
CORPUS_COLUMNS = ("title", "context")


def passage_document_id(passage: str) -> str:
    """Derive a stable document id from passage text.

    :param passage: Passage text, verbatim.
    :type passage: str
    :returns: ``"squad-"`` plus the first 16 hex chars of the passage's SHA-1.
    :rtype: str
    """
    return "squad-" + hashlib.sha1(passage.encode("utf-8")).hexdigest()[:16]


def build_documents(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn SQuAD rows into unique, label-free ingestion documents.

    Reads only ``title`` and ``context`` from each row; any other key
    (``question``, ``answers``, ``id``) is ignored. SQuAD repeats each passage
    once per question, so passages are de-duplicated, then sorted by id so the
    output order is stable.

    :param rows: SQuAD records, or anything with ``title`` and ``context`` keys.
    :type rows: Iterable[Mapping[str, Any]]
    :returns: Documents in the ``POST /corpus/documents`` shape.
    :rtype: list[dict[str, Any]]
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        passage = row["context"]
        doc_id = passage_document_id(passage)
        by_id.setdefault(
            doc_id,
            {"id": doc_id, "text": passage, "metadata": {"title": row["title"], "source": SOURCE}},
        )
    return [by_id[doc_id] for doc_id in sorted(by_id)]


def _call(method: str, url: str, body: Any = None) -> dict[str, Any]:
    """Send one JSON request and return the decoded JSON response.

    :param method: HTTP method.
    :type method: str
    :param url: Absolute URL.
    :type url: str
    :param body: JSON-serialisable request body, or ``None``.
    :type body: Any
    :returns: The decoded response body.
    :rtype: dict[str, Any]
    :raises SystemExit: On a non-2xx response or connection failure.
    """
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"{method} {url} -> HTTP {error.code}: {error.read().decode()}") from None
    except urllib.error.URLError as error:
        raise SystemExit(f"{method} {url} failed: {error.reason}. Is the service running?") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="service base URL")
    parser.add_argument("--batch-size", type=int, default=200, help="documents per request")
    parser.add_argument("--reset", action="store_true", help="empty the corpus first")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    from datasets import load_dataset

    print(f"Loading {DATASET_NAME} [{SPLIT}] columns {CORPUS_COLUMNS} ...")
    rows = load_dataset(DATASET_NAME, split=SPLIT).select_columns(list(CORPUS_COLUMNS))
    documents = build_documents(rows)
    print(f"{len(rows)} rows -> {len(documents)} unique passages")

    if args.reset:
        reset = _call("DELETE", f"{base}/corpus")
        print(f"Reset: deleted {reset['documentsDeleted']} documents, {reset['chunksDeleted']} chunks")

    totals = dict.fromkeys(
        ("documentsCreated", "documentsReplaced", "documentsUnchanged", "chunksWritten", "chunksDeleted"), 0
    )
    started = time.perf_counter()
    corpus: dict[str, Any] = {}
    for offset in range(0, len(documents), args.batch_size):
        batch = documents[offset : offset + args.batch_size]
        result = _call("POST", f"{base}/corpus/documents", {"documents": batch})
        for key in totals:
            totals[key] += result[key]
        corpus = result["corpus"]
        print(f"  {offset + len(batch)}/{len(documents)} documents sent")
    elapsed = time.perf_counter() - started

    print(f"Done in {elapsed:.1f}s: " + ", ".join(f"{key}={value}" for key, value in totals.items()))
    print(f"Corpus now: {corpus['documents']} documents, {corpus['chunks']} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())