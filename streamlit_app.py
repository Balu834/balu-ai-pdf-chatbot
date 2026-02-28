import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with Your PDF (Stable Version)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Load model directly
    qa_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    query = st.text_input("Ask a question about your PDF:")

    if query:
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        result = qa_pipeline(prompt, max_length=512)
        answer = result[0]["generated_text"]

        st.subheader("📖 Answer")
        st.write(answer)
