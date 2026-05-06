import os
import logging
import httpx
from fastapi import APIRouter
from app.core.config import settings
from app.services.vector_store_service import PineconeVectorStore

router = APIRouter(prefix="/api/system", tags=["system"])
logger = logging.getLogger(__name__)

@router.get("/status")
async def get_system_status():
    """
    Check availability of external services and local environment.
    """
    status = {
        "directories": {},
        "pinecone": {"configured": False, "connected": False, "index_ready": False},
        "ollama": {"available": False, "models": []},
    }

    # 1. Check Directories
    for name, path in [("uploads", settings.UPLOAD_DIR), ("output", settings.OUTPUT_DIR)]:
        status["directories"][name] = {
            "path": path,
            "exists": os.path.exists(path),
            "writable": os.access(path, os.W_OK) if os.path.exists(path) else False
        }

    # 2. Check Pinecone
    if settings.PINECONE_API_KEY:
        status["pinecone"]["configured"] = True
        try:
            vstore = PineconeVectorStore()
            if vstore.has_index():
                status["pinecone"]["connected"] = True
                status["pinecone"]["index_ready"] = True
                status["pinecone"]["dimension"] = vstore.describe_dimension()
        except Exception as e:
            status["pinecone"]["error"] = str(e)

    # 3. Check Ollama
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                status["ollama"]["available"] = True
                models_data = response.json().get("models", [])
                status["ollama"]["models"] = [m.get("name") for m in models_data]
                
                status["ollama"]["embedding_model_present"] = settings.OLLAMA_EMBEDDING_MODEL in status["ollama"]["models"]
                status["ollama"]["llm_model_present"] = settings.LLM_MODEL in status["ollama"]["models"]
    except Exception as e:
        status["ollama"]["error"] = str(e)

    return status
