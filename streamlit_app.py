import streamlit as st

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Balu AI Labs", layout="wide")

# ------------------------
# SIMPLE LOGIN SYSTEM
# ------------------------
def login():

    st.title("🔐 Balu AI Labs Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "balu" and password == "balu123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

# ------------------------
# MAIN APP
# ------------------------
def main_app():

    st.sidebar.title("Balu AI Labs")
    menu = st.sidebar.radio("Navigation", ["Dashboard", "PDF Chatbot", "Logout"])

    if menu == "Dashboard":
        st.title("🚀 Welcome to Balu AI Labs")
        st.success("Login successful!")
        st.write("This is your AI platform dashboard.")

    elif menu == "PDF Chatbot":

        st.header("📄 AI PDF Chatbot (Free Version)")

        uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

        if uploaded_file is not None:

            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.chains import RetrievalQA
            from langchain_community.llms import HuggingFaceHub

            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",
                model_kwargs={"temperature":0.5, "max_length":512}
            )

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            question = st.text_input("Ask something about the document:")

            if question:
                answer = qa.run(question)
                st.markdown("### 🤖 Answer")
                st.write(answer)

        else:
            st.info("Please upload a PDF file to start chatting.")

    elif menu == "Logout":
        st.session_state.logged_in = False
        st.experimental_rerun()

# ------------------------
# SESSION CONTROL
# ------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login()
