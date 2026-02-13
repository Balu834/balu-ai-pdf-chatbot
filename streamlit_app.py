import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI PDF Chatbot (FREE)", layout="centered")
st.title("📄 AI PDF Chatbot (FREE Version)")

# -----------------------------
# File Upload (ONLY ONE)
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# -----------------------------
# Load & Build RAG
# -----------------------------
@st.cache_resource
def build_rag(pdf_path):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Free HuggingFace text generation model
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=generator)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa

# -----------------------------
# Main App Logic
# -----------------------------
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    qa = build_rag(tmp_path)

    question = st.text_input("Ask something about the document:")

    if question:
        with st.spinner("Thinking..."):
            answer = qa.run(question)

        st.markdown("### 🤖 Answer")
        st.write(answer)

else:
    st.info("Please upload a PDF file to start chatting.")
