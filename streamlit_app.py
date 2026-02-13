import streamlit as st
import os

# PDF + RAG imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline


# ----------------------------
# LOGIN SYSTEM
# ----------------------------

def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "balu123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")


# ----------------------------
# PDF CHATBOT FUNCTION
# ----------------------------

def pdf_chatbot():

    st.header("📄 AI PDF Chatbot (Free Version)")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file is not None:

        with
