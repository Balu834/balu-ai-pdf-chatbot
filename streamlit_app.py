import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI PDF Chat",
    page_icon="📄",
    layout="wide"
)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:700;
        margin-bottom:10px;
    }
    .subtitle {
        font-size:18px;
        color:gray;
        margin-bottom:30px;
    }
    .answer-box {
        background-color:#f0f2f6;
        padding:20px;
        border-radius:12px;
        font-size:16px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown('<div class="main-title">💬 AI PDF Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a PDF and ask questions instantly</div>', unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    st.markdown("---")
    st.info("This AI finds relevant content from your PDF and answers your questions.")

# ---------------------------
# Main Logic
# ---------------------------
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content

    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    st.success(f"✅ PDF processed successfully ({len(chunks)} text sections created)")

    query = st.text_input("🔎 Ask a question about your PDF")

    if query:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(chunks + [query])

        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        most_similar_index = np.argmax(similarity)

        answer = chunks[most_similar_index]

        st.markdown("### 📖 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

else:
    st.info("👈 Upload a PDF from the sidebar to begin.")
