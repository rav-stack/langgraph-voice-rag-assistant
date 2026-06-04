# IT Service Copilot (RAG)
Enterprise IT support assistant built with Retrieval-Augmented Generation (RAG).  
Knowledge files are chunked, embedded using Hugging Face, stored in Chroma, and answered through Groq Llama using retrieved context.
Remote: [https://github.com/rav-stack/IT-Service-Copilot-rag](https://github.com/rav-stack/IT-Service-Copilot-rag)
## Features
- FastAPI endpoint: `POST /ask`
- Retrieval pipeline with:
  - Chroma similarity search
  - heuristic scoring
  - LLM pointwise reranking (`YES`/`NO`)
- Grounded answer generation:
  - instructed to answer only from provided context
  - returns fallback: `"Insufficient data to process an answer"` when context is insufficient
- Source-aware responses:
  - response includes source file names used for the answer
## Project Structure
| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI app and `/ask` route |
| `app/services/ingest_service.py` | Load → chunk → embed → store pipeline |
| `app/services/retrieval_service.py` | Retrieve + rerank relevant chunks |
| `app/services/llm_service.py` | Groq answer generation |
| `app/services/vectorstore_service.py` | Chroma + embedding initialization |
| `app/utils/loaders.py` | Load `.txt` files from `data/raw/` |
| `app/utils/chunking.py` | Custom chunking logic |
| `scripts/ingest_data.py` | Ingestion entrypoint |
| `data/raw/` | Knowledge base text files |
## Tech Stack
- Python 3.10+ (3.11 recommended)
- FastAPI + Uvicorn
- LangChain ecosystem:
  - `langchain_chroma`
  - `langchain_huggingface`
  - `langchain_text_splitters`
- Chroma (persistent vector DB)
- Groq (`llama-3.1-8b-instant`)
## Prerequisites
- Python 3.10+
- Groq API key from [https://console.groq.com/](https://console.groq.com/)
## Setup
### 1) Clone repository
```bash
git clone https://github.com/rav-stack/IT-Service-Copilot-rag.git
cd IT-Service-Copilot-rag
2) Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
3) Configure environment variables
Create .env in project root:

GROQ_API_KEY=your_groq_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_DB_DIR=./chroma_db
Variable reference:

GROQ_API_KEY: required for LLM calls
EMBEDDING_MODEL: embedding model name for HuggingFace
CHROMA_DB_DIR: folder for persistent Chroma storage
Ingest Knowledge Base
Place .txt files in data/raw/, then run:

python -m scripts.ingest_data
This loads, chunks, and indexes data into Chroma.

Run API
uvicorn app.main:app --reload
App URL: http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs

API Usage
Request
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I request VPN access?"}'
Response (current implementation)
{
  "question": "How do I request VPN access?",
  "answer": "You can request VPN access by ... [vpn_troubleshooting.txt]",
  "sources": ["vpn_troubleshooting.txt"]
}
.gitignore (Important)
To avoid pushing local virtual environment binaries and large files to GitHub, ensure .gitignore includes:

#evaluation script
pytho-m scripts.evaluate

.venv/
__pycache__/
*.py[cod]
.DS_Store
chroma_db/
.env