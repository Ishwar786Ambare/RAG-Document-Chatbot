# rag_pipeline.py
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"Using Google API Key: {GEMINI_API_KEY}")

def load_and_chunk_pdf(pdf_path: str):
    """Load PDF and split into chunks"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 500 characters per chunk
        chunk_overlap=50,    # 50 char overlap between chunks
        length_function=len
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from PDF")
    return chunks

def create_vector_store(chunks):
    """Create embeddings and store in ChromaDB"""
    # Free HuggingFace model - no API key needed
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Store vectors locally in 'chroma_db' folder
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Vector store created successfully")
    return vector_store

def get_answer(vector_store, question: str) -> str:
    """Get answer from document using RAG"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    system_prompt = (
        "You are a helpful assistant. Use the following context to answer questions.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return result