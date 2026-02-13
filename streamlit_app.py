import streamlit as st

# --------------------
# SESSION STATE LOGIN
# --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --------------------
# LOGIN PAGE
# --------------------
def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# --------------------
# DASHBOARD PAGE
# --------------------
def dashboard():
    st.title("🚀 Welcome to Balu AI Labs")
    st.success("Login successful!")
    st.write("This is your AI platform dashboard.")

# --------------------
# PDF CHATBOT PAGE
# --------------------
def pdf_chatbot():
    st.title("📄 AI PDF Chatbot")
    st.info("PDF Chatbot coming next step...")

# --------------------
# MAIN APP
# --------------------
if not st.session_state.logged_in:
    login()
else:
    st.sidebar.title("Balu AI Labs")
    page = st.sidebar.radio("Navigation", ["Dashboard", "PDF Chatbot", "Logout"])

    if page == "Dashboard":
        dashboard()
    elif page == "PDF Chatbot":
        pdf_chatbot()
    elif page == "Logout":
        st.session_state.logged_in = False
        st.rerun()
