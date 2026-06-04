from app.schemas.chat import ChatResponse


def test_chat_response_has_required_fields():
    resp = ChatResponse(
        session_id="s1",
        answer="hello",
        sources=["vpn_access.md"],
        grounding_note="Answer grounded in retrieved sources.",
        latency_ms=12.3,
    )
    data = resp.model_dump()
    for key in ["session_id", "answer", "sources", "grounding_note", "latency_ms"]:
        assert key in data