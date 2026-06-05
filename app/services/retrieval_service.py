from app.services.vectorstore_service import get_vectorstore
#from app.services.llm_service import client


def retrieve_documents(query, k=8):
    vectorstore = get_vectorstore()
    
    # Step 1: retrieve more candidates (increase recall)
    results = vectorstore.similarity_search(query, k=k)

    #step 2: heuristic scoring
    query_words = query.lower().split()
    stopwords =["the", "is", "at", "a", "an", "to", "for", "on", "in", "of", "and"]

    document_score =[]
    for doc in results:
        content = doc.page_content.lower()
        #count words in doc overlapping with query
        score = sum(1 for word in query_words if word in content and word not in stopwords)

        document_score.append((doc,score))
        # Step 3: sort by score (high → low)
        
    document_score.sort(key =lambda x: x[1], reverse=True)


##causing lot of latency 

    # top_docs = [doc for doc, _ in document_score[:5]]


    # #step4 : LLm Pointwise reranking
    # final_docs =[]

    # for doc in top_docs:
    #     prompt = f"""
    #     You are a relevance evaluator.

    #     Query: {query}

    #     Document:
    #     {doc.page_content}

    #     Answer ONLY with:
    #     YES or NO
    #     """

    #     response = client.chat.completions.create(
    #         model = "llama-3.1-8b-instant",
    #         messages = [{"role" :"user", "content" : prompt}],
    #         temperature = 0
    #     )

    #     verdict = response.choices[0].message.content.strip().upper()
        
    #     if "YES" in verdict:
    #         final_docs.append(doc)

    # # for i in final_docs[:3]:
    # #     source = i.metadata.get("source","unknown")
    # #     preview= i.page_content[:100].replace("\n", " ")
    # #     print(f"\n source : {source}  content : {preview}")
    # return final_docs


    scored = [doc for doc, score in document_score if score > 0]
    top_docs = scored[:5] if scored else [doc for doc, _ in document_score[:5]]
    
    return top_docs



 



    


