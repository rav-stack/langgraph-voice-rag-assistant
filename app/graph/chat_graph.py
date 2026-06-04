from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

from app.graph.state import ChatState
from app.services.retrieval_service import retrieve_documents
from app.services.llm_service import generate_answer_with_history  


def retrieve_node(state: ChatState) -> dict:
    # The latest user message is the last item in messages object 
    question = state["messages"][-1].content

    docs = retrieve_documents(question)


    context = "\n\n".join(
        f"Source : {d.metadata.get('source')}\nContent:{d.page_content}" for d in docs
    )
    sources = list(set(d.metadata.get("source", "unknown") for d in docs))


    return {
        "retrieved_docs": context,
        "sources": sources,
        "retrieval_hits": len(docs),
    }


def generate_node(state: ChatState) -> dict:
    answer = generate_answer_with_history(
        messages=state["messages"],
        context=state["retrieved_docs"],
    )
    
    return {"messages": [AIMessage(content=answer)]}

def build_chat_graph():
    builder = StateGraph(ChatState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve") #start and go to retrieve node so that we have question
    builder.add_edge("retrieve", "generate") #retreieval to generate answer
    builder.add_edge("generate", END)  #generate and end

    return builder.compile(checkpointer=MemorySaver())


# build once at import, reuse for every request and it gets saved for each session id in RAM
chat_graph = build_chat_graph()