import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub


# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with Your PDF (AI Powered)")

# ----------------------------
# Load HuggingFace Token
# ----------------------------
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    # ----------------------------
    # Split Text into Chunks
    # ----------------------------
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # ----------------------------
    # Create Embeddings
    # ----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # ----------------------------
    # Ask Question
    # ----------------------------
    user_question = st.text_input("Ask a question about your PDF:")

    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)

        st.subheader("📖 Answer:")
        st.write(response)
