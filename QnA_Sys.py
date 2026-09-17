from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import streamlit as st
import tempfile
import os

def load_pdf_from_upload(uploaded_files):
    """Load PDFs from uploaded files"""
    documents = []
    for uploaded_file in uploaded_files:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Load the PDF
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        documents.extend(docs)
        
        # Clean up the temporary file
        os.unlink(tmp_file_path)
    
    return documents

def create_chunks_from_docs(documents):
    """Create chunks from documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(documents)

    for chunk in text_chunks:
        if not isinstance(chunk.page_content, str):
            chunk.page_content = str(chunk.page_content)

    return text_chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

@st.cache_resource
def get_vectorstore(_uploaded_files):
    """Create vectorstore with caching based on file hash"""
    documents = load_pdf_from_upload(_uploaded_files)
    text_chunks = create_chunks_from_docs(documents)
    embedding_model = get_cached_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    return db

@st.cache_resource
def get_cached_embedding_model():
    """Cache the embedding model to avoid reloading"""
    embedding_model = HuggingFaceEmbeddings(
        model = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

@st.cache_resource
def load_llm():
    llm = OllamaLLM(model="llama3.2:1b")
    return llm

def get_prompt(prompt_template, input_variables):
    prompt = PromptTemplate(template = prompt_template,
                            input_variables = input_variables)
    return prompt