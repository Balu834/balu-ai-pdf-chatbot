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

    mode = st.radio(
        "🔍 Mode",
        ["Smart Search", "Full LLM (OpenAI)"]
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.query_count = 0
        st.success("Chat cleared!")

# ---------------------------
# DARK MODE STYLING
# ---------------------------
if dark_mode:
    background_color = "#121212"
    text_color = "white"
    ai_color = "#1f1f1f"
    user_color = "#2563eb"
else:
    background_color = "white"
    text_color = "black"
    ai_color = "#f3f4f6"
    user_color = "#dbeafe"

st.markdown(f"""
<style>
body {{
    background-color: {background_color};
}}

.chat-container {{
    max-width: 850px;
    margin: auto;
    padding-bottom: 120px;
}}

.user-bubble {{
    background-color: {user_color};
    color: white;
    padding: 14px 18px;
    border-radius: 20px;
    margin-bottom: 10px;
    max-width: 70%;
    margin-left: auto;
    animation: fadeIn 0.3s ease-in;
}}

.ai-bubble {{
    background-color: {ai_color};
    color: {text_color};
    padding: 14px 18px;
    border-radius: 20px;
    margin-bottom: 10px;
    max-width: 70%;
    animation: fadeIn 0.3s ease-in;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(5px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.header-title {{
    text-align:center;
    font-size:36px;
    font-weight:700;
}}

.footer {{
    text-align:center;
    margin-top:40px;
    font-size:14px;
    color:gray;
}}

.stChatInputContainer {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: {background_color};
    padding: 10px 0;
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
    pdf_name = uploaded_file.name

    st.markdown(f"📄 **Current Document:** {pdf_name}")

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

            if mode == "Full LLM (OpenAI)" and "OPENAI_API_KEY" in st.secrets and OpenAI:
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"Answer based on this PDF content:\n{text[:4000]}"},
                        {"role": "user", "content": query}
                    ]
                )

                answer = response.choices[0].message.content

            else:
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
