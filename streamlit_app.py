import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Balu AI Labs", layout="centered")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# -------------------------
# LOGIN FUNCTION
# -------------------------
def login():
    if (
        st.session_state.username == "balu"
        and st.session_state.password == "balu123"
    ):
        st.session_state.authenticated = True
    else:
        st.error("Invalid username or password")

# -------------------------
# LOGOUT FUNCTION
# -------------------------
def logout():
    st.session_state.authenticated = False


# =========================
# LOGIN PAGE
# =========================
if not st.session_state.authenticated:

    st.title("🔐 Balu AI Labs Login")

    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")

    st.button("Login", on_click=login)

# =========================
# PROTECTED APP
# =========================
else:

    st.title("🚀 Balu AI Labs")
    st.success("Login successful!")

    st.markdown("## 📄 AI PDF Chatbot")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
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

        pipe = pipeline("text-generation", model="google/flan-t5-base")
        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        question = st.text_input("Ask something about your PDF")

        if question:
            answer = qa.run(question)
            st.write("### 🤖 Answer")
            st.write(answer)

    st.markdown("---")
    st.button("Logout", on_click=logout)
