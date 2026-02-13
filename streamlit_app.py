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

        # FREE HuggingFace model
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
