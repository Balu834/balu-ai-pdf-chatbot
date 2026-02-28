import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime

# Optional OpenAI
try:
    from openai import OpenAI
except:
    OpenAI = None

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI PDF Chat", page_icon="💬", layout="wide")

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    st.markdown("---")

    dark_mode = st.toggle("🌙 Dark Mode")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.success("Chat cleared!")

# ---------------------------
# DARK MODE STYLING
# ---------------------------
if dark_mode:
    bg_color = "#1e1e1e"
    text_color = "white"
    ai_color = "#2d2d2d"
    user_color = "#2563eb"
else:
    bg_color = "white"
    text_color = "black"
    ai_color = "#f3f4f6"
    user_color = "#dbeafe"

st.markdown(f"""
<style>
.chat-container {{
    max-width: 800px;
    margin: auto;
}}
.user-bubble {{
    background-color: {user_color};
    color: {text_color};
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 70%;
    margin-left: auto;
}}
.ai-bubble {{
    background-color: {ai_color};
    color: {text_color};
    padding: 12px 18px;
    border-radius: 18px;
    margin-bottom: 10px;
    width: fit-content;
    max-width: 70%;
}}
.header-title {{
    text-align:center;
    font-size:36px;
    font-weight:700;
}}
.footer {{
    text-align:center;
    margin-top:50px;
    font-size:14px;
    color:gray;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="header-title">💬 AI PDF Chat Assistant</div>', unsafe_allow_html=True)

# ---------------------------
# SESSION STATE
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# ---------------------------
# PDF PROCESSING
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
    # DISPLAY CHAT
    # ---------------------------
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='user-bubble'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-bubble'>{message}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------
    # CHAT INPUT
    # ---------------------------
    query = st.chat_input("Type your message...")

    if query:
        st.session_state.query_count += 1

        with st.spinner("AI is thinking..."):

            # Try OpenAI if key exists
            if "OPENAI_API_KEY" in st.secrets and OpenAI:
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Answer based on provided PDF content."},
                        {"role": "user", "content": query}
                    ]
                )

                answer = response.choices[0].message.content

            else:
                # Fallback: Similarity Search
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(chunks + [query])

                similarity = cosine_similarity(vectors[-1], vectors[:-1])
                most_similar_index = np.argmax(similarity)
                answer = chunks[most_similar_index]

        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("ai", answer))
        st.rerun()

else:
    st.info("👈 Upload a PDF to begin chatting.")

# ---------------------------
# DOWNLOAD CONVERSATION
# ---------------------------
if st.session_state.chat_history:
    conversation_text = ""
    for role, message in st.session_state.chat_history:
        conversation_text += f"{role.upper()}: {message}\n\n"

    st.download_button(
        label="📥 Download Conversation",
        data=conversation_text,
        file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# ---------------------------
# USAGE STATS
# ---------------------------
st.markdown("---")
st.markdown(f"📊 Total Questions Asked: **{st.session_state.query_count}**")

# ---------------------------
# BRANDING
# ---------------------------
st.markdown('<div class="footer">Built by Balu AI Labs 🚀</div>', unsafe_allow_html=True)
