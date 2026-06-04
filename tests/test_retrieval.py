import os
import pytest


@pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1",
                    reason="integration test; set RUN_INTEGRATION=1 to run")
def test_retrieval_returns_something():
    from app.services.retrieval_service import retrieve_documents
    docs = retrieve_documents("How do I request VPN access?")
    assert len(docs) > 0