# phase2_rag_chain.py

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS   # was: Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def load_vector_db(path="faiss_index"):  # was: "chroma_index"
    """Load Chroma vector database."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )

def create_rag_chain():
    """Construct a RetrievalQA chain for HealthHER use cases."""
    vector_db = load_vector_db()
    # Basic retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are HealthHER AI — a sensitive, knowledgeable assistant for women’s physical, mental, reproductive health, as well as parenting and legal support.
Use the context to answer succinctly, kindly, and ask for region if not provided.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
