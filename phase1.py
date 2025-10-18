# phase1_ingest.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def load_documents(folder_path: str):
    """Load all text or PDF documents from a folder, returning list of Documents."""
    docs = []
    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(full_path)
        elif fname.lower().endswith(".txt"):
            loader = TextLoader(full_path, encoding="utf-8")
        else:
            continue
        docs.extend(loader.load())
    return docs

def chunk_documents(docs):
    """Split large documents into smaller chunks for better embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def create_vector_store(docs, save_path="chroma_index"):  # was: "faiss_index"
    """Create and save Chroma vector store using OpenAI embeddings."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,                 # in newer langchain, param name is ⁠ embedding ⁠
        persist_directory=save_path
    )
    vectorstore.persist()
    print("✅ Vector store created and saved at:", save_path)

if __name__ == "__main__":
    folder = "data"
    raw_docs = load_documents(folder)
    print(f"Loaded {len(raw_docs)} documents.")
    chunks = chunk_documents(raw_docs)
    print(f"Created {len(chunks)} chunks.")
    create_vector_store(chunks)

