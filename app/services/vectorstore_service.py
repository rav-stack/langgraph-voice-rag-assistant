import os
from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name = os.getenv("EMBEDDING_MODEL"))

    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMA_DB_DIR"),
        embedding_function=embedding_model
    )

    return vectorstore

###test input

# vectorstore= get_vectorstore()
# vectorstore.add_texts(["Rag improves accuracy"])
# results = vectorstore.similarity_search("What is RAG?")

# print(results)



