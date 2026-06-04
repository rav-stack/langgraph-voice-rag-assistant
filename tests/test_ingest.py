import os
import tempfile
from app.utils.loaders import load_documents


def test_ingest_reads_a_file():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "note.txt")
        open(path, "w").write("hello world")

        docs = load_documents(folder)

        assert len(docs) == 1                
        assert docs[0]["source"] == "note.txt"