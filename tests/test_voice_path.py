from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_voice_needs_audio():
    r = client.post("/api/voice/chat")        # no audio sent

    assert r.status_code == 422               # route exists and rejects missing audio