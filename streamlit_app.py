import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline


# ---------------- LOGIN SYSTEM ---------------- #

def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "balu123":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")


def logout():
    st.session_state["logged_in"] = False
    st.rerun()


# ---------------- DASHBOARD ---------------- #

def dashboard():
    st.title("🚀 Welcome to Balu AI Labs")
    st.success("Login successful!")
    st.write("This is your AI platform dashboard.")


# ---------------- PDF CHATBOT ---------------- #

def pdf_chatbot():
    st.title("📄 AI PDF Chatbot (Free Version)")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:

        # Save file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)

        # FREE local LLM
        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        question = st.text_input("Ask something about the document:")

        if question:
            answer = qa.run(question)
            st.write("### 🤖 Answer")
            st.write(answer)


# ---------------- MAIN APP ---------------- #

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
            logout()


if __name__ == "__main__":
    main()
