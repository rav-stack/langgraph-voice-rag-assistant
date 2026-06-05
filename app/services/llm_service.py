# app/services/llm_service.py

import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Single source of truth for the "can't answer" message.
# Used in the prompt AND by the endpoint to detect ungrounded answers.
GROUNDING_FALLBACK = "I do not have enough information in the knowledge base."

def generate_answer(query, context):
    prompt = f"""
You are an enterprise IT support assistant.

Answer ONLY using the provided context.
If the answer is not present, say: "Insufficient data to process an answer".

Also ensure the answer is concise and accurate.
While answering, you MUST cite the exact source file name from the context.
Use this format: [source_file_name]

Do NOT invent or modify source names.
Only use sources explicitly provided in the context.
Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )


HISTORY_WINDOW = 6  # how many recent messages to send to the LLM


def _to_groq_messages(messages):
    """Convert LangChain message objects to Groq's {role, content} dicts."""
    role_map = {"human": "user", "ai": "assistant"}
    return [
        {"role": role_map.get(m.type, "user"), "content": m.content}
        for m in messages
    ]


def generate_answer_with_history(messages, context):
    # System message: same grounding rules as generate_answer, sent once
    system_prompt = {
          "role": "system",
          "content": (
              "You are an enterprise IT support assistant. "
              "Answer ONLY using the provided context. "
              f'If the answer is not present, say exactly: "{GROUNDING_FALLBACK}" '
              "Be concise and accurate. You MUST cite the exact source file name "
              "from the context using the format [source_file_name]. "
              "Do NOT invent or modify source names."
          ),
      }

    # The retrieved context, as its own system message
    context_msg = {
        "role": "system",
        "content": f"Context:\n{context}",
    }

    # Recent conversation history, converted to Groq format
    history = _to_groq_messages(messages[-HISTORY_WINDOW:])

    full_messages = [system_prompt, context_msg, *history]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=full_messages,
        temperature=0,
    )

    return response.choices[0].message.content.strip()    













# import os
# from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def generate_answer(query, context):
#     prompt = f"""
#  You are an enterprise support assistant.

#  Answer ONLY from the given context.
#  If the answer is not in the context, say "I don't know".

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

#     response = client.chat.completions.create(
#         model="gpt-4.1-mini",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.2
#     )

#     return response.choices[0].message.content.strip()