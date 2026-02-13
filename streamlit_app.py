import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="Balu AI Labs", layout="wide")

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "balu123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
def dashboard():
    st.title("🚀 Welcome to Balu AI Labs")
    st.success("Login successful!")
    st.write("This is your AI platform dashboard.")

# ---------------- PDF CHATBOT ----------------
def pdf_chatbot():
    st.title("📄 AI PDF Chatbot (Free Version)")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(texts, embeddings)

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

        query = st.text_input("Ask something about your PDF:")

        if query:
            result = qa.run(query)
            st.write("### 🤖 Answer")
            st.write(result)

# ---------------- MAIN ----------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
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
            st.session_state.logged_in = False
            st.rerun()

if __name__ == "__main__":
    main()
