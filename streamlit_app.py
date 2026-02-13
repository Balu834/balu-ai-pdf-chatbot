import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Balu AI Labs", layout="centered")

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN FUNCTION ----------------
def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- MAIN APP ----------------
def main_app():
    st.title("📄 Balu AI Labs - AI PDF Chatbot")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        st.success("PDF uploaded successfully!")
        st.write("Chatbot logic will go here.")

# ---------------- ROUTING ----------------
if st.session_state.logged_in:
    main_app()
else:
    login()
