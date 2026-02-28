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
    page_icon="💬",
    layout="wide"
)

# ---------------------------
# Custom Styling
# ---------------------------
st.markdown("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
}
.user-bubble {
    background-color: #dbeafe;
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 10px;
    text-align: right;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
}
.ai-bubble {
    background-color: #f3f4f6;
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 70%;
}
.header-title {
    text-align:center;
    font-size:36px;
    font-weight:700;
}
.sub-text {
    text-align:center;
    color:gray;
    margin-bottom:30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown('<div class="header-title">💬 AI PDF Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload a PDF and chat instantly</div>', unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

# ---------------------------
# Initialize Chat Memory
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------
# PDF Processing
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

    st.success(f"✅ PDF processed successfully ({len(chunks)} sections created)")

    # ---------------------------
    # Display Chat
    # ---------------------------
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(
                f"<div class='user-bubble'><b>You:</b><br>{message}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='ai-bubble'><b>AI:</b><br>{message}</div>",
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # Input at Bottom
    # ---------------------------
    query = st.text_input("Type your message...")

    if query:
        with st.spinner("AI is thinking..."):
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(chunks + [query])

            similarity = cosine_similarity(vectors[-1], vectors[:-1])
            most_similar_index = np.argmax(similarity)
            answer = chunks[most_similar_index]

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))

        st.rerun()

else:
    st.info("👈 Upload a PDF from the sidebar to begin chatting.")
