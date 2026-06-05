# Architecture Notes

IT Service Copilot — a chat + voice knowledge assistant over a small IT-support
knowledge base, with grounded answers, multi-turn memory, offline evaluation, and
observability.

---

## 1. System Overview

```text
                        ┌──────────────────────────┐
                        │   Client / curl / Web UI │
                        └────────────┬─────────────┘
                                     │
        ┌────────────┬───────────────┼───────────────┬────────────┐
        ▼            ▼               ▼               ▼            ▼
   ┌─────────┐ ┌───────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐
   │ /api/   │ │ /api/     │ │ /api/voice/  │ │ /api/    │ │ /api/    │
   │ health  │ │ ingest    │ │ chat         │ │ chat     │ │ evaluate │
   └─────────┘ └─────┬─────┘ └──────┬───────┘ └────┬─────┘ └────┬─────┘
                     │              │              │            │
                     │              ▼              │            │
                     │       ┌─────────────┐       │            │
                     │       │ STT         │       │            │
                     │       │ (faster-    │       │            │
                     │       │  whisper)   │       │            │
                     │       └──────┬──────┘       │            │
                     │              │              │            │
                     │              ▼              ▼            ▼
                     │        ┌──────────────────────────────────┐
                     │        │      LangGraph  chat_graph        │
                     │        │  ┌──────────┐    ┌────────────┐   │
                     │        │  │ retrieve │ ─▶ │  generate  │   │
                     │        │  └────┬─────┘    └─────┬──────┘   │
                     │        │       │                │          │
                     │        │   MemorySaver (per-session memory)│
                     │        └───────┼────────────────┼──────────┘
                     │                ▼                ▼
                     │        ┌──────────────┐  ┌──────────────┐
                     │        │ Retrieval    │  │ LLM Service  │
                     │        │ (Chroma +    │  │ (Groq /      │
                     │        │  rerank)     │  │  Llama 3.1)  │
                     │        └──────┬───────┘  └──────────────┘
                     ▼               ▼
              ┌──────────────┐ ┌──────────────┐
              │  data/raw    │ │   Chroma     │
              │ txt·md·pdf   │▶│ Vector Store │
              └──────────────┘ └──────────────┘
```

> Note: `/api/ingest/upload` (multi-file upload) saves files into `data/raw` then runs
> the same ingest path shown above.

The chat and voice endpoints share **one** pipeline: voice simply transcribes audio to
text, then enters the exact same LangGraph chat flow. Ingestion populates the vector
store; evaluation replays a fixed dataset through the chat graph.

---

## 2. Ingestion & Retrieval

```text
  INGESTION (write path)
  ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐
  │ data/raw   │ ▶ │ Loaders      │ ▶ │ Chunking     │ ▶ │ Embeddings │ ▶ │ Chroma   │
  │ txt·md·pdf │   │ (by ext)     │   │ (splitter)   │   │ (MiniLM)   │   │ (store)  │
  └────────────┘   └──────────────┘   └──────────────┘   └────────────┘   └──────────┘

  RETRIEVAL (read path)
  ┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ User       │ ▶ │ Similarity   │ ▶ │ Heuristic    │ ▶ │ LLM pointwise│ ▶ │ Context +    │
  │ question   │   │ search top-k │   │ lexical score│   │ rerank YES/NO│   │ sources      │
  └────────────┘   └──────┬───────┘   └──────────────┘   └──────────────┘   └──────────────┘
                          ▲
                   ┌──────┴───────┐
                   │   Chroma     │
                   └──────────────┘
```

**Ingestion.** Files are read by type — `txt`/`md` as plain text, `pdf` via `pypdf`.
The loader keeps a stable `{content, source}` contract so chunking and indexing never
change when new formats are added. Re-ingest is a refresh: existing vectors are cleared
first, so the store never accumulates duplicates.

**Retrieval.** Three stages: vector similarity search for recall, a lexical
word-overlap score to re-order, and an LLM pointwise relevance filter (YES/NO per doc)
for precision. Each chunk carries its `source` filename, which flows into the prompt and
the response for citations.

---

## 3. Chat Flow (LangGraph)

```text
  ┌──────────────────────┐
  │ message + session_id │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────────────┐        ┌─────────────────────┐
  │ chat_graph.invoke            │ ◀────▶ │ MemorySaver         │
  │ thread_id = session_id       │  load/ │ checkpointer        │
  └──────────┬───────────────────┘  save  │ (per-session memory)│
             ▼                  history    └─────────────────────┘
  ┌──────────────────────────────────────────┐
  │  LangGraph (state machine)               │
  │                                          │
  │   start ─▶ ┌──────────┐ ─▶ ┌──────────┐ ─▶ end
  │           │ retrieve  │    │ generate │   │
  │           │ node      │    │ node     │   │
  │           └──────────┘    └────┬─────┘   │
  └────────────────────────────────┼─────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ answer · sources · grounding_note ·  │
                │ latency_ms                           │
                └──────────────────────────────────────┘
```

**State.** A typed `ChatState` flows through the graph. Its `messages` field uses the
`add_messages` reducer, so each turn **appends** to history instead of overwriting.

**Memory.** The `thread_id` (= `session_id`) keys the `MemorySaver` checkpointer, which
auto-loads and saves per-session history. Same session id → remembered; new id → fresh
conversation. Retrieval runs on the latest question only; history shapes generation.

**Why LangGraph over plain LangChain.** State is first-class, the checkpointer gives
multi-turn memory for free, and new nodes/branches (e.g. query reformulation) can be
added without rewriting control flow. The assignment also required it to be visible in
code.

---

## 4. Voice Flow

```text
  ┌──────────────┐   ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────────┐
  │ audio +      │ ▶ │ temp     │ ▶ │ faster-      │ ▶ │ transcript │ ▶ │ same         │ ▶ │ transcript · │
  │ session_id   │   │ file     │   │ whisper      │   │            │   │ chat_graph   │   │ answer ·     │
  │ (multipart)  │   │          │   │ transcribe   │   │            │   │              │   │ sources      │
  └──────────────┘   └──────────┘   └──────────────┘   └────────────┘   └──────────────┘   └──────────────┘
```

Audio is saved to a temp file (Whisper reads from a path), transcribed locally with
faster-whisper, then fed into the **same** chat graph as text — inheriting memory,
retrieval, and grounding. The model is lazy-loaded once and reused. Text chat works with
no voice configured, since the STT model only loads on first voice use.

**Why faster-whisper (open-source) over Deepgram.** No API key or signup, runs locally
and offline, reviewer-friendly, and CPU/int8 defaults run on any machine.

---

## 5. Grounding & Error Handling

- **Grounding.** The system prompt instructs the model to answer only from retrieved
  context and to return a fixed fallback — *"I do not have enough information in the
  knowledge base."* — when evidence is missing. The endpoint detects that fallback to set
  `grounding_note`. The fallback string is a single shared constant used by both the
  prompt and the detector, so they can't drift apart.
- **Model errors.** The graph call is wrapped in try/except; failures log an ERROR record
  (with `request_id`) and return a clean HTTP 503 instead of a stack trace.
- **Voice errors.** Transcription and chat failures are caught separately and logged as
  distinct events (`voice_error` vs `voice_chat_error`).
- **Evaluation isolation.** `/api/evaluate` runs the eval in a **subprocess**, so the
  heavy ML work cannot destabilize the API server.

---

## 6. Observability

Every chat/voice request logs a structured line via a shared logger:
`request_id`, `session_id`, `latency_ms`, `retrieval_hits` (and `transcript_len` for
voice). `request_id` identifies a single request; `session_id` groups a conversation.
`/api/health` reports application, LLM, vector-store, and voice readiness, plus an
explicit `knowledge_base_ingested` flag and chunk count.

---

## 7. Testing

Five focused tests, one per required area, kept fast and deterministic:

| Area | Test | Approach |
|---|---|---|
| Ingestion | `test_ingest.py` | Load a temp folder; confirm files become `{content, source}` |
| Retrieval | `test_retrieval.py` | Real `retrieve_documents` returns chunks — opt-in (`RUN_INTEGRATION=1`) |
| Chat format | `test_chat_format.py` | Validate the `ChatResponse` schema has all required fields |
| Voice path | `test_voice_path.py` | `POST /api/voice/chat` with no audio → 422 (route wired, no model load) |
| Eval runner | `test_eval_runner.py` | Dataset has the required 10+ examples and all three types |

Heavy end-to-end retrieval is gated behind an env flag, because running the full ML
stack (PyTorch + tokenizers + Groq) inside the pytest harness triggers the same native
fork/OpenMP instability. The default `pytest` run stays fast and crash-free; integration
is run deliberately. An empty `conftest.py` at the root puts the project on the import path.

---

## 8. Key Tradeoffs

| Decision | Choice (this build) | Production direction |
|---|---|---|
| Conversation memory | In-memory `MemorySaver` (wiped on restart) | Durable checkpointer (Postgres/Redis) |
| Reranking | LLM pointwise (precise, slow) | Local cross-encoder / batching |
| Re-ingestion | Full wipe-and-reload | Incremental sync (changed files only) |
| Evaluation | Deterministic checks (explainable) | Add LLM-judge layer for nuanced relevance |
| `/api/evaluate` | Synchronous subprocess | Background job queue + job id |
| OpenMP conflict | `KMP_DUPLICATE_LIB_OK` workaround | Single-OpenMP env (conda) |
| `get_vectorstore` | Re-instantiated per call | Cached (load-once) like the graph/STT |

---

## 9. Stack

FastAPI · LangGraph · LangChain · Chroma · HuggingFace embeddings (MiniLM) ·
Groq (Llama 3.1) · faster-whisper · Pydantic. Configuration via environment variables;
no secrets committed.
