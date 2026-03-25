# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from app.services.rag_pipeline import (
    load_and_chunk_pdf,
    create_vector_store,
    get_answer
)

app = FastAPI(title="RAG Document Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global vector store (in production use Redis/DB)
vector_store = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and create vector store"""
    global vector_store
    
    # Save uploaded file
    pdf_path = f"./uploads/{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF → chunks → vectors
    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = create_vector_store(chunks)
    
    return {
        "message": "PDF processed successfully",
        "chunks_created": len(chunks),
        "filename": file.filename
    }

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Ask question about uploaded document"""
    global vector_store
    
    if vector_store is None:
        return {"error": "Please upload a PDF first"}
    
    answer = get_answer(vector_store, request.question)
    
    return {
        "question": request.question,
        "answer": answer
    }

@app.get("/health")
async def health():
    return {"status": "running"}























# Run with: uvicorn main:app --reload

# from fastapi import FastAPI
# from .routers import chat, embeddings

# app = FastAPI(title="AI API", description="FastAPI app with AI/GenAI services")

# app.include_router(chat.router)
# app.include_router(embeddings.router)

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the AI API"}
