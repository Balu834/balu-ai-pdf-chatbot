import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with Your PDF")

# Load HuggingFace token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    # LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    query = st.text_input("Ask a question about your PDF:")

    if query:
        result = qa_chain.run(query)
        st.subheader("📖 Answer")
        st.write(result)
