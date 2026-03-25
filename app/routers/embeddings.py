import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..models.request import DocumentIngestRequest
from ..models.response import EmbeddingResponse, DocumentIngestResponse
from ..services.vector_store import VectorStoreService
from pypdf import PdfReader

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])

vector_store = VectorStoreService()

@router.post("/", response_model=EmbeddingResponse)
async def create_embeddings(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if vector_store.store is None:
        raise HTTPException(status_code=500, detail="Vector store is not configured")

    count = vector_store.add_documents([text])
    embeddings = vector_store.embeddings.embed_query(text) if vector_store.embeddings else []

    return {"embeddings": embeddings}

@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(payload: DocumentIngestRequest):
    if not payload.text:
        raise HTTPException(status_code=400, detail="Document text is required")

    inserted = vector_store.add_documents([payload.text], metadatas=[payload.metadata or {}])
    return {"success": True, "inserted_count": inserted}

@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    if not pages:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    inserted = vector_store.add_documents(pages)
    return {"success": True, "inserted_count": inserted}
