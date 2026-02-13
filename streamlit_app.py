import streamlit as st

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Balu AI Labs", layout="centered")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# -------------------------
# LOGIN FUNCTION
# -------------------------
def login():
    USERNAME = "balu"
    PASSWORD = "balu123"

    if (
        st.session_state.username == USERNAME
        and st.session_state.password == PASSWORD
    ):
        st.session_state.authenticated = True
    else:
        st.error("Invalid username or password")

# -------------------------
# LOGOUT FUNCTION
# -------------------------
def logout():
    st.session_state.authenticated = False


# -------------------------
# LOGIN PAGE
# -------------------------
if not st.session_state.authenticated:
    st.title("🔐 Balu AI Labs Login")

    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")

    st.button("Login", on_click=login)

# -------------------------
# MAIN APP (Protected Area)
# -------------------------
else:
    st.title("🚀 Welcome to Balu AI Labs")

    st.success("Login successful!")

    st.write("This area is protected.")
    st.write("Your AI apps will appear here.")

    st.button("Logout", on_click=logout)
