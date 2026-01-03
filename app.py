import os
import streamlit as st
from typing_extensions import List, TypedDict
from dotenv import load_dotenv
import tempfile

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

st.set_page_config(page_title="PDF RAG App", layout="wide", page_icon=":books:")
st.title("Chat with multiple PDFs :books:")

st.sidebar.header("Upload files Here!")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# llm=ChatOllama(
#     model="llama3",
#     temperature=0
#     )
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)
prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable AI assistant.

Use ONLY the information provided in the context below.
Explain clearly and in detail.
Write at least 4–6 sentences.
Do NOT give one-word answers.

Context:
{context}

Question:
{question}

Detailed Answer:
""")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Processing PDFs and building vector store..."):

        all_documents = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_pdf_path = tmp.name

            loader = PyPDFLoader(temp_pdf_path)

            docs = loader.load()
            all_documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

        all_splits = text_splitter.split_documents(all_documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(all_splits, embeddings)

        st.session_state.vector_store = vector_store

        st.success("PDFs are processed and indexed successfully")

def get_answer(question: str) -> str:
    vector_store = st.session_state.vector_store
    retrived_docs = vector_store.similarity_search(question)

    docs_content = "\n\n".join(doc.page_content for doc in retrived_docs)

    messages = prompt.invoke({
        "question": question,
        "context": docs_content
    })

    response = llm.invoke(messages)
    return response.content

st.markdown("---")
st.subheader("Ask a question")

question = st.text_input("Enter your question based on the uploaded PDFs")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload PDF files first...")
    else:
        with st.spinner("Generating..."):
            answer = get_answer(question)

        st.markdown("### Answer")
        st.write(answer)