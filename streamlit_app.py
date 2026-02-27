import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="PDF Reader", page_icon="📄")

st.title("📄 Simple PDF Reader")
st.write("Upload a PDF to view its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    total_pages = len(reader.pages)

    st.success(f"Total Pages: {total_pages}")

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.subheader("PDF Content Preview:")
    st.write(text[:2000])