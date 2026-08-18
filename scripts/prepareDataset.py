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
    """
    folded = unicodedata.normalize("NFKD", question)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = _PUNCTUATION.sub(" ", folded.lower())
    return _WHITESPACE.sub(" ", folded).strip()


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """SQuAD validation records carry several annotator answers, often identical."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def group_by_title(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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

    Records already matching a used question key or a used context are skipped,
    which keeps topics spread across articles instead of clustering.
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

    Question and context are copied verbatim - never reformatted or summarized.
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