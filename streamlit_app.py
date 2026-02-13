import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Balu AI Labs", layout="wide")

# -----------------------------
# SIMPLE LOGIN SYSTEM
# -----------------------------
def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "balu123":
            st.session_state["logged_in"] = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# -----------------------------
# DASHBOARD
# -----------------------------
def dashboard():
    st.title("🚀 Welcome to Balu AI Labs")
    st.success("Login successful!")
    st.write("This is your AI platform dashboard.")

# -----------------------------
# FREE PDF CHATBOT
# -----------------------------
def pdf_chatbot():
    st.title("📄 AI PDF Chatbot (Free Version)")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

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

        # FREE LLM (small model)
        pipe = pipeline(
            "text-generation",
            model="distilgpt2",
            max_length=200
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        question = st.text_input("Ask a question about your PDF:")

        if question:
            answer = qa.run(question)
            st.markdown("### 🤖 Answer")
            st.write(answer)

# -----------------------------
# MAIN APP
# -----------------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        login()
    else:
        st.sidebar.title("Balu AI Labs")
        menu = st.sidebar.radio(
            "Navigation",
            ["Dashboard", "PDF Chatbot", "Logout"]
        )

        if menu == "Dashboard":
            dashboard()

        elif menu == "PDF Chatbot":
            pdf_chatbot()

        elif menu == "Logout":
            st.session_state["logged_in"] = False
            st.rerun()


if __name__ == "__main__":
    main()
