from fastapi import FastAPI
from pydantic import BaseModel
from app.services.retrieval_service import retrieve_documents
from app.services.llm_service import generate_answer, GROUNDING_FALLBACK
import os
from app.services.ingest_service import ingest_data
import shutil
from fastapi import UploadFile, File, HTTPException,Form
import time
import uuid
import tempfile
from langchain_core.messages import HumanMessage
from app.graph.chat_graph import chat_graph
from app.schemas.chat import ChatRequest,ChatResponse,VoiceChatResponse
from app.utils.logger import logger
from app.services.stt_service import transcribe_audio
import subprocess
import sys
import glob
import json


DATA_PATH = os.getenv("DATA_PATH","data/raw")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/api/ingest")
def ingest_data_endpoint():
    return ingest_data(DATA_PATH)

@app.post("/api/ingest/upload")
async def ingest_upload_endpoint(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        dest_path = os.path.join(DATA_PATH, file.filename)
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)

    # re-run ingestion once, after all files are saved
    result = ingest_data(DATA_PATH)
    result["uploaded_files"] = saved
    return result

@app.post("/ask")
def ask_question(request: QueryRequest):
    docs = retrieve_documents(request.query)

    #context = "\n".join([doc.page_content for doc in docs]) 
    # #LLM hallucinated because source names because metadata was not propagated into the prompt.
    # the below function passess source name as well as the content of the chunk to the LLM

    context = "\n\n".join([f"Source : {doc.metadata.get('source')}\nContent:{doc.page_content}" for doc in docs])

    sources = list(set([doc.metadata.get("source","unknown") for doc in docs]))
    answer = generate_answer(request.query,context)
    
    return {
        "question" : request.query,
        "answer": answer,
        "sources" : sources
        
    }

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request :ChatRequest):

    session_id = request.session_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    start = time.perf_counter()
    #thread_id = session_id in memory checkpoint
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = chat_graph.invoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "session_id": session_id,
            },
            config=config,
        )
    except Exception as e:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            f"chat_error | request_id={request_id} session_id={session_id} "
            f"latency_ms={latency_ms} error={type(e).__name__}: {e}"
        )
        raise HTTPException(
            status_code=503,
            detail="The assistant is temporarily unavailable. Please try again.",
        )

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    answer = result["messages"][-1].content
    sources = result.get("sources", [])
    retrieval_hits = result.get("retrieval_hits", 0)

    # grounding_note: answer from documents or fallback
    if GROUNDING_FALLBACK.lower() in answer.lower():
        grounding_note = "No sufficient evidence found in the knowledge base."
    else:
        grounding_note = "Answer grounded in retrieved sources."


    logger.info(
        f"chat | request_id={request_id} session_id={session_id} "
        f"latency_ms={latency_ms} retrieval_hits={retrieval_hits}"
           )    

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources,
        grounding_note=grounding_note,
        latency_ms=latency_ms,
    )    


@app.post("/api/voice/chat", response_model=VoiceChatResponse)
def voice_chat_endpoint(
    audio: UploadFile = File(...),
    session_id: str | None = Form(default=None),
):
    session_id = session_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    # 1. Save the uploaded audio to a temp file (Whisper reads from a path)
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    # 2. Transcribe (with its own error handling — voice errors are logged)
    try:
        transcript = transcribe_audio(tmp_path)
    except Exception as e:
        logger.error(
            f"voice_error | request_id={request_id} session_id={session_id} "
            f"error={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=503, detail="Transcription failed.")
    finally:
        os.remove(tmp_path)  # clean up the temp file either way

    # 3. Feed the transcript into the SAME chat graph
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = chat_graph.invoke(
            {"messages": [HumanMessage(content=transcript)], "session_id": session_id},
            config=config,
        )
    except Exception as e:
        logger.error(
            f"voice_chat_error | request_id={request_id} session_id={session_id} "
            f"error={type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable.")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    answer = result["messages"][-1].content
    sources = result.get("sources", [])
    retrieval_hits = result.get("retrieval_hits", 0)

    if GROUNDING_FALLBACK.lower() in answer.lower():
        grounding_note = "No sufficient evidence found in the knowledge base."
    else:
        grounding_note = "Answer grounded in retrieved sources."

    logger.info(
        f"voice_chat | request_id={request_id} session_id={session_id} "
        f"latency_ms={latency_ms} retrieval_hits={retrieval_hits} "
        f"transcript_len={len(transcript)}"
    )

    return VoiceChatResponse(
        session_id=session_id,
        transcript=transcript,
        answer=answer,
        sources=sources,
        grounding_note=grounding_note,
        latency_ms=latency_ms,
    )

@app.post("/api/evaluate")
def evaluate_endpoint():
    # Run eval in a separate process so heavy ML doesn't destabilize the server
    proc = subprocess.run(
        [sys.executable, "-m", "eval.run"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {proc.stderr[-500:]}",
        )

    # Read the most recent JSON report that eval.run just wrote
    reports = sorted(glob.glob("eval/reports/report_*.json"))
    if not reports:
        raise HTTPException(status_code=500, detail="No report generated.")

    latest = reports[-1]
    with open(latest) as f:
        report = json.load(f)

    return {"summary": report["summary"], "report_path": latest}

#uvicorn app.main:app --reload
