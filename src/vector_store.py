from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

def create_vectorstore(documents):
    embeddings = FakeEmbeddings(size=1536)
    vectorstore = Chroma.from_documents(documents, embeddings, persist_directory='data/vectorstore')
    return vectorstore

def load_vectorstore(path):
    embeddings = FakeEmbeddings(size=1536)
    return Chroma(persist_directory=path, embedding_function=embeddings)

def save_vectorstore(vectorstore, path):
    pass