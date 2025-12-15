import streamlit as st
from src.ingestion import load_documents
from src.vector_store import create_vectorstore, save_vectorstore, load_vectorstore
from src.llm import create_qa_chain
import os
from dotenv import load_dotenv

load_dotenv()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

st.title("AI Business Assistant")

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])

if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        path = os.path.join('data', uploaded_file.name)
        with open(path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(path)
    
    documents = load_documents(file_paths)
    if documents:
        st.session_state.vectorstore = create_vectorstore(documents)
        save_vectorstore(st.session_state.vectorstore, 'data/vectorstore')
        st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
        st.success("Documents processed!")
    else:
        st.error("No valid documents uploaded.")

elif os.path.exists('data/vectorstore'):
    st.session_state.vectorstore = load_vectorstore('data/vectorstore')
    st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)

if st.session_state.qa_chain:
    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain(question)
        st.write("**Answer:**", answer)
else:
    st.info("Please upload documents to get started.")