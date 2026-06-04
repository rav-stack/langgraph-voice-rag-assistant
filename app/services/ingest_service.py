
# this service will run when we trigger the ingest_data.py

from app.utils.loaders import load_documents
from app.utils.chunking import chunk_documents
from app.services.vectorstore_service import get_vectorstore


def ingest_data(data_path):
    #step 1 : load documents
    documents = load_documents(data_path)

    #step2 : chunk document
    chunks = chunk_documents(documents)

    #step 3: store vector db
    vectorstore = get_vectorstore()

#step 4: refresh - clear existing vectors so re-ingest doesn't duplicate
    existing = vectorstore.get()
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])

    texts =[obj["content"] for obj in chunks]
    metadatas = [{"source" : obj["source"]} for obj in chunks]

    vectorstore.add_texts(texts=texts,metadatas=metadatas)
    
#step 5: report counts for the API response
    result = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "status": "success",
    }
    
    print(f"✅ Ingestion complete: {result}")
    return result




    