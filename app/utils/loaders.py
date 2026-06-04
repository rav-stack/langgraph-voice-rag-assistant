import os
from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf"}


def _load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    except Exception as e:
        print(f"Failed to read PDF {file_path}: {e}")
        return ""


def load_documents(data_path):
    documents = []

    for file in os.listdir(data_path):

        file_path = os.path.join(data_path, file)

        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file)[1].lower()

        if ext not in SUPPORTED:
            print(f"Skipping unsupported file: {file}")
            continue

        if ext in {".txt", ".md"}:
            text = _load_text(file_path)
        else:
            text = _load_pdf(file_path)

        if text.strip():
            documents.append(
                {
                    "content": text,
                    "source": file
                }
            )

    return documents


# # ✅ Test input
# data_path = "data/raw"
# result = load_documents(data_path)
# print (result)

