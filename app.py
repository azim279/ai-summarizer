import streamlit as st
import fitz  # PyMuPDF
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import os

# Set your OpenAI API key
import os
import streamlit as st

# Load OpenAI API key from Streamlit Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Page config
st.set_page_config(page_title="AI Research Paper Summarizer", layout="centered")
st.title("📄 AI Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload a PDF research paper", type="pdf")

if uploaded_file is not None:
    st.info("Reading the PDF file...")
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join([page.get_text() for page in doc])

    if len(text.strip()) == 0:
        st.error("Couldn't extract any text from the PDF. Please try a different file.")
    else:
        raw_docs = [Document(page_content=text)]

        st.info("Summarizing with GPT-4...")
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        chain = load_summarize_chain(llm, chain_type="stuff")
        summary = chain.run(raw_docs)

        st.success("Done! Here's your summary:")
        st.text_area("📚 Summary", summary, height=300)

        st.download_button("💾 Download Summary as TXT", summary, file_name="summary.txt")
