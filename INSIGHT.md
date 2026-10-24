# Project insights

Plain-language notes on how this project works and why it is built this way.
For commands and rules, see `README.md` and `CLAUDE.md`.

## The big picture

```
golden dataset (60 questions + expected answers)      <- stays on the test side
        |
        | question only
        v
Playwright test --HTTP--> rag_service --search--> corpus (ChromaDB)
                               |
                               +--> Groq LLM writes an answer from the retrieved passages
```

- **RAG** (Retrieval-Augmented Generation) means: first *retrieve* relevant text, then let an
  LLM *generate* an answer from that text only.
- **Golden dataset** (`data/golden/rag_test_cases.json`): 60 SQuAD questions with known answers.
  50 are answerable and 10 are not, so the service should abstain on those 10.
- **Corpus**: the 1,204 Wikipedia passages that the service searches, stored in ChromaDB.
- **Framework vs. system under test**: the Playwright/TypeScript side is the test framework.
  `rag_service/` is only a sample app for it to test. The two talk over HTTP only, so the same
  tests can point at any other RAG app.

## Vectors in ChromaDB, from zero

### 1. What is a vector?

A vector is a list of numbers, for example `[0.12, -0.40, 0.88, ...]`. In this project every
vector has **384 numbers**.

### 2. Where do the numbers come from? (embeddings)

An **embedding model** reads a piece of text and outputs a vector. This project uses
`all-MiniLM-L6-v2`. The model was trained so that **texts with similar meaning get similar
numbers**:

| Text | Vector (made-up, shortened) |
|---|---|
| "When did the Shah's regime collapse?" | `[0.81, 0.10, -0.33, ...]` |
| "The Shah's regime began to collapse in January 1979." | `[0.78, 0.14, -0.30, ...]` |
| "Photosynthesis happens in chloroplasts." | `[-0.52, 0.66, 0.21, ...]` |

The first two share almost no exact wording, but their meaning is close, so their numbers are
close. The third is about something else, so its numbers are far away. This is why vector search
finds answers by **meaning**, not by keyword.

Each of the 384 numbers on its own means nothing you can name. Only the whole pattern means
something.

### 3. Think of it as a map

Picture each vector as a point on a map. With 2 numbers you get a flat map; with 384 you get a
map with 384 directions that you cannot draw but that works the same way:

```
          ^
          |    * "Shah regime collapse?" (question)
          |   * "regime collapsed in 1979" (passage)   <- close = related
          |
          |                        * "photosynthesis..."  <- far = unrelated
          +------------------------------------------>
```

Searching means: drop the question on the map and pick the passages nearest to it.

### 4. How "near" is measured: cosine similarity

This project measures closeness by the **angle** between two vectors (cosine), not by straight-
line distance (`rag_service/app/corpus.py` creates the collection with `"space": "cosine"`).

- ChromaDB returns a **cosine distance**: `0` = same direction (same meaning), up to `2` =
  opposite direction.
- The API turns that into **similarity = 1 − distance** (`rag_service/app/main.py`), so higher
  means more related. You see this as `similarity` on each piece of `evidence` in the tests.

### 5. What ChromaDB does

ChromaDB is a **vector database**: it stores each chunk of text next to its vector and can
quickly find the nearest vectors to a new one.

- **On ingest** (`POST /corpus/documents`): split the passage into chunks, embed each chunk into
  a vector, and store the text, vector and metadata (`title`, `source`).
- **On search** (`POST /retrieve`): embed the question with the *same* model, then ask ChromaDB
  for the `topK` nearest chunks (default 4).
- Comparing the question with every stored vector would be slow on large corpora, so ChromaDB
  uses an index called **HNSW**, a kind of shortcut graph that finds the near neighbours
  without checking every one.
- The data lives on disk in `rag_service/.chroma/`, so it survives restarts.

### 6. Why the same model must be used everywhere

Vectors from different models live on different "maps"; comparing them is meaningless. That is
why the collection records its embedding model and refuses to open under a different one.

### 7. The embedding model, and why it is needed

**What it is.** `all-MiniLM-L6-v2` is a small neural network (6 layers, ~80 MB) built for one job:
turn a sentence or paragraph into 384 numbers that capture its meaning. It does not write or
answer anything. Here it runs **locally**, through ChromaDB's bundled ONNX build
(`rag_service/app/embeddings.py`), so it is free and needs no API key. PyTorch is avoided because
it has no current wheels for Intel Macs. The model files are downloaded once to `~/.cache/chroma`.

**Why it is needed.** A computer cannot compare the *meaning* of two texts directly. The options
without embeddings are poor:

| Approach | Problem |
|---|---|
| Keyword search | Misses paraphrases: "Shah's regime collapse" vs. "the monarchy fell" share no words |
| Give the LLM the whole corpus | 1,204 passages are far too long for one prompt, and slow and costly per question |
| Embeddings + vector search | Finds text by meaning, in milliseconds, before the LLM is ever called |

So the embedding model is the **"find"** step and the Groq LLM is the **"write"** step:

```
question --embedding model--> vector --ChromaDB nearest search--> top 4 chunks
                                                                       |
                                       Groq LLM reads only these --> answer or abstain
```

**Two models, two jobs:**

| | Embedding model | LLM (Groq) |
|---|---|---|
| Model | `all-MiniLM-L6-v2` | `openai/gpt-oss-120b` (default) |
| Runs | Locally, inside `rag_service` | Groq's cloud, needs `GROQ_API_KEY` |
| Input → output | Text → 384 numbers | Question + chunks → answer text |
| Used by | Ingest and every search | `/query` only |

**Limits to know.**
- It reads at most **256 tokens** (~1,000 characters) and silently drops the rest. That is the
  reason for chunking.
- It is trained mostly on English. Questions in other languages will match poorly.
- It is small and fast, not the most accurate. A weak retrieval result can be the embedding
  model's fault, not the LLM's. The `stage` field on errors and the separate `/retrieve`
  endpoint help tell these apart.

### 8. Other embedding models, and why we kept MiniLM

**Groq has no embedding model for this project.** Groq's model list for our key (checked
2026-10-14) contains only chat LLMs, guard classifiers and speech models. So embeddings stay local,
or would come from another provider (Google Gemini, Jina and Voyage have free tiers, but each
adds an API key and a network call to every search).

**Stronger free models that run locally without PyTorch** (for example through `fastembed`,
which uses ONNX the way ChromaDB does now):

| Model | Vector size | Max input | Compared with MiniLM |
|---|---|---|---|
| `all-MiniLM-L6-v2` (current) | 384 | 256 tokens | Baseline: small and fast |
| `BAAI/bge-small-en-v1.5` | 384 | 512 tokens | Usually better retrieval, about the same size; the easiest upgrade |
| `BAAI/bge-base-en-v1.5` | 768 | 512 tokens | Better still, about 4× slower |
| `nomic-embed-text-v1.5` | 768 | 8,192 tokens | Long input, so most passages wouldn't need splitting |

**Decision (2026-10-14): keep `all-MiniLM-L6-v2` for now.** Without a retrieval hit-rate
metric there's no way to prove another model is better. That metric asks, for each golden
question, whether its passage is in the top 4 results. Build the metric first, then compare
models by its number.

**What switching would take:**
- A new `Embedder` in `rag_service/app/embeddings.py`.
- Allowing the new name in the `EMBEDDING_MODEL` setting.
- Re-ingesting the whole corpus, because the collection is locked to its embedding model, and
  vectors from different models cannot be compared.
- Possibly raising `CHUNK_SIZE`, since newer models read 512 tokens or more.

### 9. Who does what: embedding model, ChromaDB, LLM

A common shortcut is "the embedding model retrieves the passages". It doesn't do the
search itself. Three parts share the work:

1. **Embedding model:** turns text into vectors. Each chunk is embedded once at ingest, and each
   question when it is asked.
2. **ChromaDB:** stores the chunk vectors and finds the ones nearest to the question's vector.
   This is the actual retrieval.
3. **LLM (Groq):** reads only those nearest chunks and writes the answer, or abstains.

In short: the embedding model translates text into numbers, ChromaDB searches them, and the LLM
writes the answer.

### 10. PyTorch, ONNX and ChromaDB

**PyTorch** is a Python library for building, training and running neural networks. Most AI
models, MiniLM included, are created and trained with it.
- The usual way to run MiniLM is the `sentence-transformers` package, which needs PyTorch.
- **Why it isn't used here:** PyTorch stopped publishing builds for Intel Macs after 2.2.2, and
  current `sentence-transformers` needs a newer one, so it won't install on this machine. It is
  also hundreds of MB, while this project only needs to *run* a model, not train one.

**ONNX** (Open Neural Network Exchange) is a standard file format for a trained model. A `.onnx`
file is like a model "exported" from PyTorch: the network's structure and learned numbers, with no
training machinery.
- **ONNX Runtime** is a small, fast engine that runs `.onnx` files, with no PyTorch needed.
- **Here:** ChromaDB ships MiniLM already converted to ONNX and runs it with ONNX Runtime, so we
  get the same model without PyTorch. The files are downloaded once to `~/.cache/chroma`.
- **The catch:** ONNX Runtime 1.24 and later also dropped Intel Mac builds, which is why
  `rag_service/requirements.txt` pins `onnxruntime<1.24` on Intel Macs only.

**ChromaDB** is a vector database. It keeps text, its vector and its metadata together and
answers "which stored vectors are nearest to this one?".
- **Storage:** it runs embedded inside the service, with no separate server, and saves to disk at
  `rag_service/.chroma/`.
- **Search:** it uses an **HNSW** index for fast nearest-neighbour search, set to compare
  vectors by cosine.
- **Embeddings:** its bundled embedding function supplies the ONNX MiniLM.
- **Records:** each chunk is stored as `<documentId>#<chunkIndex>`, with a fingerprint that
  makes re-ingesting the same text a no-op.

```
               ┌──────────────── rag_service (Python) ────────────────┐
text ──────────► MiniLM (.onnx) run by ONNX Runtime ──► 384 numbers   │
               │                                          │           │
               │                              ChromaDB stores/searches│
               │                                          │ top 4     │
               └──────────────────────────────────────────┼───────────┘
                                                          ▼
                                           Groq LLM (cloud) ──► answer
PyTorch: not used (needed for training; no current Intel Mac builds)
```

## Project-specific details worth knowing

- **Chunking (1,000 characters, 200 overlap).** MiniLM reads at most 256 tokens and silently
  ignores the rest. Long passages are therefore split into overlapping chunks so every part is
  searchable: 1,204 passages become 1,473 chunks. The overlap stops a sentence from being cut in
  half and lost between two chunks.
- **Chunks are not whole passages.** A retrieved chunk can be part of a golden passage. Tests
  match on the chunk's parent `documentId`, never on text equality.
- **Stable ids.** A passage's id is `squad-` + the first 16 hex characters of the SHA-1 of its
  text. Python (ingest script) and TypeScript (`passageDocumentId`) compute the same id, so the
  tests know which document should come back for a question without the service ever reading the
  golden data.
- **Idempotent ingest.** Sending the same passage again returns `unchanged` and re-embeds nothing.
  That is why the test setup can "ingest" the 60 golden passages before every run safely.
- **No answer leakage.** Questions, answers and answerable labels never go into the corpus. If they
  did, search would find the answer word for word and every test would pass meaninglessly.
- **Retrieval and generation are separate.** `/retrieve` shows what the search found (no LLM, free,
  deterministic). `/query` adds the Groq answer. When a test fails, the error's `stage`
  (`retrieval` or `generation`) tells you which half broke.
- **Abstention.** For unanswerable questions the model is told to reply with a fixed sentinel
  text; the service turns that into `abstained: true` and `answer: null`.
- **Empty corpus is an error (409), not an abstention.** Otherwise an unanswerable test would
  "pass" against an empty database.
- **Swappable client.** Tests talk to `RagClient`, not to `rag_service` directly.
  `RagServiceClient` is the adapter for this service; another RAG app only needs another adapter.

## Glossary

| Term | Meaning |
|---|---|
| Embedding | Turning text into a vector; also the vector itself |
| Vector | List of numbers (384 here) that captures a text's meaning |
| Cosine similarity | How closely two vectors point the same way; 1 = same meaning |
| Chunk | A piece of a passage, at most ~1,000 characters |
| Top-k | How many nearest chunks a search returns |
| Evidence | The chunks retrieval returned for a question |
| Grounded answer | An answer written only from the evidence |
| Abstain | The model says the evidence does not contain the answer |
| ChromaDB | Vector database that stores chunks with their vectors and finds nearest ones |
| HNSW | ChromaDB's index for fast nearest-neighbour search |
| ONNX | Portable file format for a trained model; ONNX Runtime runs it |
| PyTorch | Library for building and training neural networks; not used here |
