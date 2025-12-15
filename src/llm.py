from langchain_openai import ChatOpenAI
import os

def create_qa_chain(vectorstore):
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0
    )
    retriever = vectorstore.as_retriever()
    
    def qa(question):
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)
        return response.content
    
    return qa