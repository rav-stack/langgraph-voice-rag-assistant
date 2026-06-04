import os
import json
import time
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage
from app.graph.chat_graph import chat_graph
from app.services.llm_service import GROUNDING_FALLBACK

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def ask(question, session_id):
    """Send one question to the chat graph, return answer + sources + latency."""
    config = {"configurable": {"thread_id": session_id}}
    start = time.perf_counter()
    result = chat_graph.invoke(
        {"messages": [HumanMessage(content=question)], "session_id": session_id},
        config=config,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return result["messages"][-1].content, result.get("sources", []), latency_ms


def run_evaluation():
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    results = []
    for ex in dataset:
        session_id = f"eval-{ex['id']}-{uuid.uuid4().hex[:6]}"

        # follow-ups: ask the setup question first in the same session
        if ex.get("setup_question"):
            ask(ex["setup_question"], session_id)

        answer, sources, latency_ms = ask(ex["question"], session_id)
        answer_lower = answer.lower()

        if ex["unsupported"]:
            passed = GROUNDING_FALLBACK.lower() in answer_lower
            checks = {"unsupported_handled": passed}
        else:
            source_correct = ex["expected_source"] in sources
            keyword_match = all(k.lower() in answer_lower for k in ex["expected_keywords"])
            checks = {"source_correct": source_correct, "keyword_match": keyword_match}

        results.append({
            "id": ex["id"],
            "type": ex["type"],
            "question": ex["question"],
            "answer": answer,
            "sources": sources,
            "expected_source": ex["expected_source"],
            "latency_ms": latency_ms,
            "checks": checks,
        })

    summary = summarize(results)
    report_path = write_report(results, summary)
    return {"summary": summary, "report_path": report_path}


def summarize(results):
    def pass_rate(check_name):
        vals = [r["checks"][check_name] for r in results if check_name in r["checks"]]
        return round(sum(vals) / len(vals), 2) if vals else None

    latencies = [r["latency_ms"] for r in results]
    return {
        "total": len(results),
        "source_correct_rate": pass_rate("source_correct"),
        "keyword_match_rate": pass_rate("keyword_match"),
        "unsupported_handled_rate": pass_rate("unsupported_handled"),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
    }


def write_report(results, summary):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(os.path.join(REPORTS_DIR, f"report_{ts}.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    md_path = os.path.join(REPORTS_DIR, f"report_{ts}.md")
    with open(md_path, "w") as f:
        f.write("# Evaluation Report\n\n")
        f.write("## Summary\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Per-question results\n\n")
        for r in results:
            f.write(f"### Q{r['id']} ({r['type']}) — {r['question']}\n")
            f.write(f"- Answer: {r['answer']}\n")
            f.write(f"- Sources: {r['sources']} (expected: {r['expected_source']})\n")
            f.write(f"- Latency (ms): {r['latency_ms']}\n")
            f.write(f"- Checks: {r['checks']}\n\n")

    return md_path


if __name__ == "__main__":
    out = run_evaluation()
    print(json.dumps(out["summary"], indent=2))
    print(f"\nReport: {out['report_path']}")