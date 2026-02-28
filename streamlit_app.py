import streamlit as st
from supabase import create_client
from PyPDF2 import PdfReader
import io

# -----------------------------
# 🔐 SUPABASE CONNECTION
# -----------------------------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# 🔐 LOGIN SYSTEM
# -----------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:

    st.title("🔐 Login to AI PDF Chat")
    email = st.text_input("Enter your email")

    if st.button("Login"):
        response = supabase.table("users").select("*").eq("email", email).execute()

        if response.data:
            st.session_state.user = response.data[0]
        else:
            supabase.table("users").insert({
                "email": email,
                "is_pro": False,
                "questions_used": 0
            }).execute()

            new_user = supabase.table("users").select("*").eq("email", email).execute()
            st.session_state.user = new_user.data[0]

        st.rerun()

    st.stop()

# -----------------------------
# 🏠 MAIN APP
# -----------------------------

user = st.session_state.user

st.sidebar.write(f"👤 Logged in as: {user['email']}")
st.sidebar.write(f"📊 Questions used: {user['questions_used']}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

st.title("🤖 AI PDF Chat Assistant")

# -----------------------------
# 🚫 FREE LIMIT CHECK
# -----------------------------

if not user["is_pro"] and user["questions_used"] >= 3:
    st.warning("🚀 Free limit reached. Upgrade to Pro for unlimited access.")
    st.link_button("Upgrade to Pro", "YOUR_STRIPE_LINK_HERE")
    st.stop()

# -----------------------------
# 📄 PDF UPLOAD
# -----------------------------

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

pdf_text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        pdf_text += page.extract_text()

    st.success("PDF processed successfully!")

# -----------------------------
# 💬 CHAT INPUT
# -----------------------------

query = st.chat_input("Type your message...")

if query and pdf_text:

    with st.chat_message("user"):
        st.write(query)

    # SIMPLE SEARCH (Replace with real LLM later)
    if query.lower() in pdf_text.lower():
        answer = "📄 Found relevant content in document."
    else:
        answer = pdf_text[:500]  # preview text

    with st.chat_message("assistant"):
        st.write(answer)

    # -----------------------------
    # 📊 UPDATE USAGE IN DATABASE
    # -----------------------------

    supabase.table("users").update({
        "questions_used": user["questions_used"] + 1
    }).eq("email", user["email"]).execute()

    st.session_state.user["questions_used"] += 1
