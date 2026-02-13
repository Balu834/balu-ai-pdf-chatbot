import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Balu AI PDF Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About This App")
st.sidebar.write("""
This AI tool allows you to:
- Upload any PDF
- Ask questions
- Get instant answers
- Generate summaries

Built by Balu 🚀
""")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# ---------------- FUNCTION TO CREATE QA ----------------
def create_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever()

    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retr
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Balu AI PDF Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About This App")
st.sidebar.write("""
This AI tool allows you to:
- Upload any PDF
- Ask questions
- Get instant answers
- Generate summaries

Built by Balu 🚀
""")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# ---------------- FUNCTION TO CREATE QA ----------------
def create_qa(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.from_documents(texts, embeddings)

    retriever = vectorstore.as_retriever()

    hf_pipeline = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa

# ---------------- CLEAR CHAT ----------------
if st.button("Clear Chat"):
    st.session_state.messages = []

# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    with st.spinner("Processing your document... Please wait..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        qa = create_qa("temp.pdf")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask something about the document...")

    if user_input:
        response = qa.run(user_input)

        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("bot", response))

    for role, message in st.session_state.messages:
        if role == "user":
            with st.chat_message("user"):
                st.write(message)
        else:
            with st.chat_message("assistant"):
                st.write(message)

    # ---------------- DOWNLOAD SUMMARY ----------------
    if st.button("Download Summary"):
        summary = qa.run("Give a short summary of this document.")
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
