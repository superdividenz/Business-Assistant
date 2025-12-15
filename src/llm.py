from langchain_openai import OpenAI

def create_qa_chain(vectorstore):
    llm = OpenAI(temperature=0)
    retriever = vectorstore.as_retriever()
    
    def qa(question):
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return llm(prompt)
    
    return qa