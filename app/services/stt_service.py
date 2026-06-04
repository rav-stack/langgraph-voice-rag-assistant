import os

# Mac/OpenMP fix: PyTorch and ctranslate2 each bundle OpenMP; allow the duplicate.
# Must be set BEFORE importing faster_whisper.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from faster_whisper import WhisperModel

# Load the model once, lazily, and reuse it (like the graph — build once)
_model = None


def _get_model():
    global _model
    if _model is None:
        model_size = os.getenv("WHISPER_MODEL", "base")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(file_path: str) -> str:
    #taking audio from path specified in file path and returning stt
    model = _get_model()
    segments, _info = model.transcribe(file_path)
    # transcribe returns segments (chunks of speech); join their text
    text = " ".join(segment.text for segment in segments).strip()
    return text