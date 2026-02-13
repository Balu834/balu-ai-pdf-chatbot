import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Balu AI Labs", layout="wide")

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "balu123":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
def dashboard():
    st.title("🚀 Welcome to Balu AI Labs")
    st.success("Login successful!")
    st.write("This is your AI platform dashboard.")

# ---------------- PDF TOOL ----------------
def pdf_tool():
    st.title("📄 AI PDF Tool (Free Version)")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        reader = PdfReader(uploaded_file)
        full_text = ""

        for page in reader.pages:
            full_text += page.extract_text()

        st.success("PDF loaded successfully!")

        query = st.text_input("Search inside your PDF:")

        if query:
            if query.lower() in full_text.lower():
                st.write("### 🔎 Found Results:")
                st.write(full_text)
            else:
                st.warning("No match found.")

# ---------------- MAIN ----------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
    else:
        st.sidebar.title("Balu AI Labs")
        menu = st.sidebar.radio(
            "Navigation",
            ["Dashboard", "PDF Tool", "Logout"]
        )

        if menu == "Dashboard":
            dashboard()

        elif menu == "PDF Tool":
            pdf_tool()

        elif menu == "Logout":
            st.session_state.logged_in = False
            st.rerun()

if __name__ == "__main__":
    main()
