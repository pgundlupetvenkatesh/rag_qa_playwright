#!/usr/bin/env python3
"""Build the normalized golden RAG evaluation dataset from SQuAD 2.0.

Reads the ``validation`` split of ``rajpurkar/squad_v2`` via the Hugging Face
``datasets`` library and writes ``data/golden/rag_test_cases.json``.

Selection is deterministic: the same inputs always produce the same 60 cases in
the same order. Diversity comes from round-robin sampling across SQuAD article
titles rather than taking a consecutive slice of the split.
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

DATASET_NAME = "rajpurkar/squad_v2"
SPLIT = "validation"
SOURCE = "squad_v2"

# Fixed seed: the per-title shuffle must be reproducible across runs/machines.
SEED = 20240611

ANSWERABLE_COUNT = 50
UNANSWERABLE_COUNT = 10

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "data" / "golden" / "rag_test_cases.json"

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_question(question: str) -> str:
    """Fold a question to a comparison key used for near-duplicate detection.

    Lowercases, strips accents and punctuation, and collapses whitespace, so
    "Who was Beyonce's father?" and "who was beyonce s father" collide.

    The key is used only by :func:`select_round_robin` to decide whether a
    candidate is a near-copy of a question already selected. It is never
    stored; the question text written to the dataset is the verbatim
    original.

    The folding is deliberately lossy (``resume`` and ``résumé`` collide, as
    do ``Who's`` and ``Whos``), which is the right trade-off for duplicate
    detection but makes it unsuitable for any semantic comparison. Because
    the key feeds the seeded selection, changing this function changes which
    cases are selected and breaks byte-for-byte reproducibility.

    :param question: Question text as it appears in SQuAD.
    :type question: str
    :returns: Lowercase, accent-free, punctuation-free key with single spaces.
    :rtype: str
    """
    folded = unicodedata.normalize("NFKD", question)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _PUNCTUATION.sub(" ", folded.lower())
    return _WHITESPACE.sub(" ", folded).strip()


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """Remove repeated strings while keeping the first occurrence in place.

    SQuAD validation records carry several annotator answers, often identical.
    :func:`to_test_case` uses this to collapse them into the case's
    ``groundTruth`` list.

    A plain ``list(set(values))`` would also de-duplicate but does not
    guarantee order, and ``groundTruth`` order is part of the committed JSON
    that must reproduce byte-for-byte. Comparison is exact: unlike
    :func:`normalize_question`, nothing is lowercased or stripped, so answers
    differing only by case or punctuation are both kept as accepted variants.

    :param values: Answer strings in annotator order.
    :type values: Iterable[str]
    :returns: The distinct strings, in order of first appearance.
    :rtype: list[str]
    """
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def group_by_title(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket SQuAD records by the Wikipedia article they came from.

    :func:`select_round_robin` takes one record per title per pass, so the
    selected cases spread across topics instead of clustering in the few
    articles with the most questions. This builds the per-title queues it
    draws from.

    Records keep their original relative order within each bucket, and the
    title order follows first appearance in ``records``. Neither is relied on
    for determinism: :func:`select_round_robin` sorts titles and sorts each
    bucket by record id before shuffling.

    :param records: Raw SQuAD records, each carrying a ``title`` key.
    :type records: Iterable[dict[str, Any]]
    :returns: Mapping from article title to the records under it.
    :rtype: dict[str, list[dict[str, Any]]]
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["title"]].append(record)
    return grouped


def select_round_robin(
    grouped: dict[str, list[dict[str, Any]]],
    quota: int,
    rng: random.Random,
    used_questions: set[str],
    used_contexts: set[str],
) -> list[dict[str, Any]]:
    """Take one record per title per pass until the quota is filled.

    Titles are visited in sorted order. Each title's bucket is sorted by record
    id and then shuffled with ``rng`` so the seeded shuffle sees identical input
    on every run; that sort is what makes selection reproducible. On each lap
    the first acceptable record from every title is taken, which spreads cases
    across articles instead of clustering in the ones with the most questions.

    A record is skipped, and discarded for good, when its
    :func:`normalize_question` key is already in ``used_questions`` or its
    context passage is already in ``used_contexts``. Both sets are mutated in
    place. :func:`main` passes the same sets and the same ``rng`` to the
    answerable and unanswerable passes, so the second pass cannot reuse a
    passage or a near-duplicate question claimed by the first. Changing the
    seed, the quota, the sort keys, or the order of the two passes changes the
    committed dataset.

    :param grouped: Records bucketed by article title, as returned by
        :func:`group_by_title`.
    :type grouped: dict[str, list[dict[str, Any]]]
    :param quota: Number of records to select.
    :type quota: int
    :param rng: Seeded generator used to shuffle each title's bucket.
    :type rng: random.Random
    :param used_questions: Normalized question keys already claimed. Updated
        in place.
    :type used_questions: set[str]
    :param used_contexts: Context passages already claimed. Updated in place.
    :type used_contexts: set[str]
    :returns: Exactly ``quota`` records in selection order.
    :rtype: list[dict[str, Any]]
    :raises RuntimeError: If every bucket is exhausted or fully de-duplicated
        before ``quota`` records have been selected.
    """
    titles = sorted(grouped)
    queues: dict[str, list[dict[str, Any]]] = {}
    for title in titles:
        # Sort before shuffling so the RNG sees a stable input ordering.
        queue = sorted(grouped[title], key=lambda record: record["id"])
        rng.shuffle(queue)
        queues[title] = queue

    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        made_progress = False
        for title in titles:
            if len(selected) >= quota:
                break
            queue = queues[title]
            while queue:
                record = queue.pop(0)
                question_key = normalize_question(record["question"])
                if question_key in used_questions or record["context"] in used_contexts:
                    continue
                used_questions.add(question_key)
                used_contexts.add(record["context"])
                selected.append(record)
                made_progress = True
                break
        if not made_progress:
            break

    if len(selected) < quota:
        raise RuntimeError(
            f"Only {len(selected)} of {quota} requested cases could be selected "
            "after de-duplication."
        )
    return selected


def to_test_case(record: dict[str, Any], index: int, answerable: bool) -> dict[str, Any]:
    """Map a raw SQuAD record onto the normalized schema.

    This is the single point where the Python side commits to the contract
    declared by ``RagTestCase`` in ``src/models/ragTestCase.ts``. The key
    names emitted here (``groundTruth``, ``squadId``, ``context`` as a list)
    are checked by name in the TypeScript loader, so renaming one breaks
    every downstream consumer.

    Question and context are copied verbatim - never reformatted or summarized.
    The context is wrapped in a single-element list because retrieval
    evaluation deals in multiple passages; SQuAD supplies exactly one.

    ``groundTruth`` is empty if and only if ``answerable`` is false. The flag
    is supplied by the caller, which knows which pool the record came from;
    the answers are never inspected to infer it. For answerable records the
    annotator answers are de-duplicated with :func:`dedupe_preserving_order`.

    :param record: Raw SQuAD record with its native keys (``id``, ``title``,
        ``question``, ``context``, ``answers``).
    :type record: dict[str, Any]
    :param index: One-based position in the final dataset, used to build the
        zero-padded ``RAG-NNNN`` id.
    :type index: int
    :param answerable: ``False`` for SQuAD 2.0 unanswerable questions.
    :type answerable: bool
    :returns: A JSON-serialisable dict in the ``RagTestCase`` shape.
    :rtype: dict[str, Any]
    """
    ground_truth = dedupe_preserving_order(record["answers"]["text"]) if answerable else []
    return {
        "id": f"RAG-{index:04d}",
        "question": record["question"],
        "context": [record["context"]],
        "groundTruth": ground_truth,
        "answerable": answerable,
        "source": SOURCE,
        "metadata": {
            "squadId": record["id"],
            "title": record["title"],
            "split": SPLIT,
        },
    }


def main() -> int:
    print(f"Loading {DATASET_NAME} [{SPLIT}] ...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"Loaded {len(dataset)} records.")

    answerable_pool: list[dict[str, Any]] = []
    unanswerable_pool: list[dict[str, Any]] = []
    for record in dataset:
        pool = answerable_pool if record["answers"]["text"] else unanswerable_pool
        pool.append(record)
    print(
        f"Pool sizes -> answerable: {len(answerable_pool)}, "
        f"unanswerable: {len(unanswerable_pool)}"
    )

    # A single shared RNG plus shared used-* sets means the unanswerable pass
    # also avoids contexts already claimed by the answerable pass.
    rng = random.Random(SEED)
    used_questions: set[str] = set()
    used_contexts: set[str] = set()

    answerable = select_round_robin(
        group_by_title(answerable_pool), ANSWERABLE_COUNT, rng, used_questions, used_contexts
    )
    unanswerable = select_round_robin(
        group_by_title(unanswerable_pool), UNANSWERABLE_COUNT, rng, used_questions, used_contexts
    )

    test_cases: list[dict[str, Any]] = []
    for record in answerable:
        test_cases.append(to_test_case(record, len(test_cases) + 1, answerable=True))
    for record in unanswerable:
        test_cases.append(to_test_case(record, len(test_cases) + 1, answerable=False))

    # Fail before writing rather than emitting a malformed golden file.
    if len(test_cases) != ANSWERABLE_COUNT + UNANSWERABLE_COUNT:
        raise RuntimeError(f"Expected 60 cases, built {len(test_cases)}.")
    if any(case["answerable"] and not case["groundTruth"] for case in test_cases):
        raise RuntimeError("An answerable case has no ground-truth answer.")
    if any(not case["answerable"] and case["groundTruth"] for case in test_cases):
        raise RuntimeError("An unanswerable case has ground-truth answers.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(test_cases, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    titles = {case["metadata"]["title"] for case in test_cases}
    print(f"Wrote {len(test_cases)} cases to {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"  answerable   : {sum(case['answerable'] for case in test_cases)}")
    print(f"  unanswerable : {sum(not case['answerable'] for case in test_cases)}")
    print(f"  distinct titles: {len(titles)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())