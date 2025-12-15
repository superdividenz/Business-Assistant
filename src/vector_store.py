from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def load_vectorstore(path):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings)

def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)