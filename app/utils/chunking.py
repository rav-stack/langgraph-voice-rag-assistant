from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_if_needed(current_chunk, source, splitter):
    if len(current_chunk)>300:
        split_chunks = splitter.split_text(current_chunk)
        return [{"content" : con, "source" :source} for con in split_chunks]
    else:
        return [{"content": current_chunk, "source": source}]


def chunk_documents(documents):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 30)

    for doc in documents:
        content = doc["content"]
        #split in sections
        sections = content.split("\n\n")

        for section in sections:
            section.strip()
            if not section:
                continue

            #check for steps 
            lines = section.split("\n")
            current_chunk = ""

            for line in lines:
                line.strip()

                if line.startswith(tuple(f"{i}." for i in range (1,20))):
                #save previous chunk
                    if current_chunk:
                        chunks.extend(split_if_needed(current_chunk,doc["source"],splitter))
                    else:
                        current_chunk=line
                else:
                    current_chunk += " "+line

            if current_chunk:
                chunks.extend(split_if_needed(current_chunk, doc["source"],splitter))
            
       
    return chunks   