# IT Service Copilot — Chat & Voice Knowledge Assistant

A multi-turn IT-support assistant with **text and voice** input. It retrieves from a
small document knowledge base (RAG), answers with **source citations**, clearly says when
it lacks evidence, and ships with **offline evaluation** and **observability**.

- **Orchestration:** LangGraph (stateful, multi-turn sessions)
- **Retrieval:** Chroma vector store + HuggingFace embeddings
- **LLM:** Groq (Llama 3.1)
- **Voice (STT):** faster-whisper (open-source, runs locally)
- **API:** FastAPI

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for design, diagrams, and tradeoffs.

---

## Prerequisites

- Python 3.11
- A [Groq](https://console.groq.com/) API key (free)
- `ffmpeg` (only for some voice audio formats): `brew install ffmpeg` (macOS) /
  `sudo apt-get install ffmpeg` (Linux)

---

## Setup

```bash
# 1. clone, then create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_DIR=./vector_store
DATA_PATH=data/raw
WHISPER_MODEL=base

# macOS stability: PyTorch + faster-whisper both bundle OpenMP
KMP_DUPLICATE_LIB_OK=TRUE
TOKENIZERS_PARALLELISM=false
```

No secrets are committed; `.env`, `.venv/`, and `vector_store/` are gitignored.

---

## Ingest the knowledge base

Add `.txt`, `.md`, or `.pdf` files to `data/raw/`, then:

```bash
python -m scripts.ingest_data
```

Or via the API once the server is running (see `/api/ingest` below).

> **First-run note:** the first ingest/query downloads and loads the embedding model,
> and the first voice request downloads/loads the Whisper model. Initial calls are slow
> (tens of seconds); subsequent calls are fast.

---

## Run the API

```bash
uvicorn app.main:app --reload
```

Default URL: `http://127.0.0.1:8000` — interactive docs at `http://127.0.0.1:8000/docs`.

---

## Endpoints & sample requests

### Health
```bash
curl -s "http://127.0.0.1:8000/api/health"
```

### Ingest (refresh from data/raw)
```bash
curl -s -X POST "http://127.0.0.1:8000/api/ingest"
```

### Ingest by upload (one or more files)
```bash
curl -s -X POST "http://127.0.0.1:8000/api/ingest/upload" \
  -F "files=@data/raw/vpn_access.md"
```

### Text chat (multi-turn)
```bash
# turn 1 — note the session_id returned
curl -s -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I request VPN access?"}'

# turn 2 — reuse the session_id for memory
curl -s -X POST "http://127.0.0.1:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How long does that approval take?", "session_id": "PASTE_ID"}'
```
Response: `session_id`, `answer`, `sources`, `grounding_note`, `latency_ms`.

### Voice chat
```bash
# generate a test clip on macOS (or supply your own audio file)
say "How do I request VPN access?" -o question.aiff

curl -s -X POST "http://127.0.0.1:8000/api/voice/chat" \
  -F "audio=@question.aiff"
```
Response: `session_id`, `transcript`, `answer`, `sources`, `grounding_note`, `latency_ms`.

### Offline evaluation
```bash
curl -s -X POST "http://127.0.0.1:8000/api/evaluate"
```
Returns a metric summary and the path to a generated report. Runs the eval in a
subprocess so it doesn't destabilize the server.

---

## Offline evaluation (CLI)

```bash
python -m eval.run
```

Runs the dataset in `eval/dataset.json` (16 examples: answerable, follow-up, and
unsupported questions) through the chat pipeline and writes timestamped JSON + Markdown
reports to `eval/reports/`. Metrics: source correctness, keyword/answer relevance,
unsupported-question handling, and latency summary, with per-question detail.

---

## Tests

```bash
pytest -q
```

Covers ingestion, retrieval, chat response format, voice path, and the eval runner. The
retrieval test is a real integration test, skipped by default; run it explicitly with:

```bash
RUN_INTEGRATION=1 pytest tests/test_retrieval.py
```

---

## Notes & known limitations

- **Conversation memory** uses LangGraph's in-memory checkpointer — it's wiped on server
  restart. Production would use a durable checkpointer (Postgres/Redis).
- **Voice is optional** — text chat works fully without any voice setup; the Whisper
  model only loads on the first voice request.
- **OpenMP** — PyTorch and faster-whisper each bundle OpenMP; `KMP_DUPLICATE_LIB_OK=TRUE`
  is the documented workaround (a single-OpenMP/conda environment is the cleaner fix).
- All configuration is via environment variables; no secrets are committed.
