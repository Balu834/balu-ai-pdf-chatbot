import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import os

st.set_page_config(page_title="Balu AI Labs - PDF Chatbot", layout="wide")

st.title("📄 AI PDF Chatbot (FREE AI Version)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:

    st.success("PDF loaded successfully!")

    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever()

    # FREE HuggingFace LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature":0, "max_length":512}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    question = st.text_input("Ask a question about your PDF")

    if question:
        answer = qa.run(question)
        st.markdown("### 🤖 Answer")
        st.write(answer)
