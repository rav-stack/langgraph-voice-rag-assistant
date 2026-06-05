# IT Service Copilot — Chat & Voice Knowledge Assistant

A multi-turn IT-support assistant with **text and voice** input. It retrieves from a
small document knowledge base (RAG), answers with **source citations**, clearly says when
it lacks evidence, and ships with **offline evaluation** and **observability**.

| | |
|---|---|
| **Orchestration** | LangGraph (stateful, multi-turn sessions) |
| **Retrieval** | Chroma vector store + HuggingFace embeddings (MiniLM) |
| **LLM** | Groq (Llama 3.1) |
| **Voice (STT)** | faster-whisper (open-source, runs locally) |
| **API / UI** | FastAPI + a single-page web UI |

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for design, diagrams, and tradeoffs.

---

## Two ways to use it (once running)

1. **Web UI** — open `http://127.0.0.1:8000/` — chat (text + voice) with an endpoint sidebar.
2. **Swagger / OpenAPI docs** — open `http://127.0.0.1:8000/docs` — try every endpoint interactively.

---

## Prerequisites

- Python 3.11
- A free [Groq](https://console.groq.com/) API key
- `ffmpeg` (for some voice audio formats): `brew install ffmpeg` (macOS) /
  `sudo apt-get install ffmpeg` (Linux)

---

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

| Variable | Purpose | Example |
|---|---|---|
| `GROQ_API_KEY` | LLM access (required) | `gsk_...` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_DB_DIR` | Vector store location on disk | `./vector_store` |
| `DATA_PATH` | Knowledge-base source folder | `data/raw` |
| `WHISPER_MODEL` | Whisper STT model size | `small.en` |

`.env`, `.venv/`, and `vector_store/` are gitignored — no secrets are committed.

---

## Knowledge base

Source documents live in **`data/raw/`**. Supported types: **`.txt`, `.md`, `.pdf`**.
Add or remove files there, then re-ingest. The sample set covers VPN access, password
policy, payroll, leave policy, and VPN troubleshooting.

**Ingest via the API** (start the server first — see below):

```bash
curl -s -X POST "http://127.0.0.1:8000/api/ingest"
```

You can also upload and ingest new files directly:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/ingest/upload" -F "files=@yourfile.pdf"
```

**Or via the CLI** (works without the server running — useful for first-time setup):

```bash
python -m scripts.ingest_data
```

Either way, re-ingesting refreshes the store (clears old vectors first, so no duplicates).

---

## Run the API

```bash
uvicorn app.main:app
```

> ⏳ **The first request is slow (expected).** The embedding model loads on the first
> query and the Whisper model loads on the first voice request (downloaded once). The
> first chat/voice call can take 15–30s; every call after is fast. This is one-time
> warm-up, not a hang.

---

## Endpoints

| Method | Path | Purpose | Output |
|---|---|---|---|
| GET | `/api/health` | App / LLM / vector-store / voice readiness | status + chunk count |
| POST | `/api/ingest` | Re-index `data/raw` | document & chunk counts |
| POST | `/api/ingest/upload` | Upload file(s) and index them | counts + uploaded files |
| POST | `/api/chat` | Text chat (multi-turn) | answer, sources, grounding_note, latency_ms |
| POST | `/api/voice/chat` | Voice chat (audio in) | transcript, answer, sources, latency_ms |
| POST | `/api/evaluate` | Run offline evaluation | metric summary + report path |

### Sample requests

```bash
# Health
curl -s "http://127.0.0.1:8000/api/health"

# Ingest (refresh from data/raw)
curl -s -X POST "http://127.0.0.1:8000/api/ingest"

# Upload & ingest
curl -s -X POST "http://127.0.0.1:8000/api/ingest/upload" \
  -F "files=@data/raw/vpn_access.md"

# Text chat — turn 1 (note the returned session_id)
curl -s -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I request VPN access?"}'

# Text chat — turn 2 (reuse session_id for memory)
curl -s -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How long does that approval take?", "session_id": "PASTE_ID"}'

# Voice chat (macOS: make a test clip with `say`)
say "How do I reset my password?" -o question.aiff
curl -s -X POST "http://127.0.0.1:8000/api/voice/chat" \
  -F "audio=@question.aiff"

# Offline evaluation
curl -s -X POST "http://127.0.0.1:8000/api/evaluate"
```

---

## Sample questions to try

**Answerable (grounded in the knowledge base):**
- How do I request VPN access?
- Which client do I use to connect to the VPN?
- How do I reset my password?
- What are the password requirements?
- When are salaries paid each month?
- How many days of maternity leave do I have?
- If my VPN keeps failing, what can I try?

**Follow-ups (test multi-turn memory — ask after the matching question):**
- "How do I request VPN access?" → then *"How long does that approval take?"*
- "What are the password requirements?" → then *"And how often do they expire?"*

**Unsupported (the assistant should refuse, not guess):**
- How do I request a new company car?
- What is the company's stock price forecast?

---

## Offline evaluation

```bash
python -m eval.run
```

Runs `eval/dataset.json` (16 examples: answerable, follow-up, unsupported) through the
chat pipeline and writes timestamped **JSON + Markdown** reports to `eval/reports/`.
Metrics: source correctness, keyword/answer relevance, unsupported-question handling, and
latency summary, with per-question detail. The same run is available over HTTP at
`POST /api/evaluate` (runs in a subprocess so it can't destabilize the server).

---

## Tests

```bash
pytest -q
```

Covers ingestion, retrieval, chat response format, voice path, and the eval runner. The
retrieval test is a real integration test, skipped by default — run it explicitly:

```bash
RUN_INTEGRATION=1 pytest tests/test_retrieval.py
```

---

## Notes & limitations

- **First request is slow** (one-time model warm-up) — see the callout above.
- **Conversation memory** uses LangGraph's in-memory checkpointer, wiped on restart.
  Production would use a durable checkpointer (Postgres/Redis).
- **Voice is optional** — text chat works fully without voice configured; the Whisper
  model only loads on the first voice request. Accuracy scales with `WHISPER_MODEL`
  (`small.en` recommended over `base`).
- **OpenMP** — PyTorch and faster-whisper each bundle OpenMP; environment flags are set at
  startup in `app/main.py` to prevent a conflict on some systems.
- All configuration is via environment variables; no secrets are committed.
