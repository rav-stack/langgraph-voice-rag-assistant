from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class ChatState(TypedDict):
    #basemessage for message objectand add_message to append messages and not overwrite
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    retrieved_docs: str      # only the context string from retrieved document object
    sources: list[str]       # source names for citations
    retrieval_hits: int      # how many chunks retrieval returned

