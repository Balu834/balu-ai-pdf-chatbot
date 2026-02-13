import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Balu AI Labs - Document Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------- HEADER ---------------- #
st.markdown(
    """
    <h1 style='text-align: center;'>🤖 Balu AI Labs</h1>
    <h3 style='text-align: center; color: gray;'>AI Document Assistant</h3>
    <hr>
    """,
    unsafe_allow_html=True
)

st.write("Upload your PDF and ask questions instantly using AI.")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("📄 Upload your PDF", type="pdf")

if uploaded_file is not None:

    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    @st.cache_resource
    def load_rag():
        loader = PyPDFLoader("temp.pdf")
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

        pipe = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_length=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        return qa

    qa = load_rag()

    question = st.text_input("💬 Ask a question about your document")

    if question:
        with st.spinner("Thinking..."):
            answer = qa.run(question)

        st.markdown("### 🤖 AI Response")
        st.write(answer)

else:
    st.info("Please upload a PDF file to start chatting.")

# ---------------- FOOTER ---------------- #
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: gray;'>
    Built with ❤️ by <b>Balu AI Labs</b><br>
    Empowering small businesses with AI
    </div>
    """,
    unsafe_allow_html=True
)
