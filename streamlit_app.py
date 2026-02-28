import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("💬 Chat with Your PDF (Stable AI Version)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    # Extract text
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    # Split into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    st.success(f"PDF Loaded Successfully. {len(chunks)} chunks created.")

    query = st.text_input("Ask a question about your PDF:")

    if query:
        # Vectorize text
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(chunks + [query])

        # Compute similarity
        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        most_similar_index = np.argmax(similarity)

        answer = chunks[most_similar_index]

        st.subheader("📖 Answer (From PDF Context)")
        st.write(answer)
