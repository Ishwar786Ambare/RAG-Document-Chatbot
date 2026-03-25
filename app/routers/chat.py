from fastapi import APIRouter, HTTPException
from ..models.request import ChatCompletionRequest
from ..models.response import ChatCompletionResponse
from ..services.llm import LLMService
from ..services.vector_store import VectorStoreService

router = APIRouter(prefix="/chat", tags=["Chat"])

vector_store = VectorStoreService()
llm_service = LLMService()

@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completion(payload: ChatCompletionRequest):
    if not payload.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result = llm_service.combined_response(payload.prompt, vector_store, top_k=payload.top_k)

    return {
        "reply": result["answer"],
        "usage": {
            "model": result["model"],
            "top_k": payload.top_k,
            "prompt_tokens": payload.max_tokens,
        },
        "retrieved_docs": result["retrieved_docs"],
    }
