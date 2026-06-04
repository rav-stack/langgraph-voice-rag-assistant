from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.services.retrieval_service import retrieve_documents
from app.services.llm_service import generate_answer

# Sample evaluation questions
data = [
    {"question": "vpn not working", "ground_truth": "troubleshoot vpn steps"}
]

questions = []
answers = []
contexts = []
ground_truths = []

for item in data:
    query = item["question"]

    docs = retrieve_documents(query)
    context = [doc.page_content for doc in docs]

    answer = generate_answer(query, "\n".join(context))

    questions.append(query)
    answers.append(answer)
    contexts.append(context)
    ground_truths.append(item["ground_truth"])

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# Run evaluation
result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    raise_exceptions=False
)

print(result)