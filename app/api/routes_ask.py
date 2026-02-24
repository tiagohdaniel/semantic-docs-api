from fastapi import APIRouter

from app.schemas.models import AskRequest, AskResponse
from app.dependencies import get_embedding_service, get_vector_store, get_llm_client
from app.services.ask_service import AskService

router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Query indexed documents")
async def ask_question(request: AskRequest):
    """Ask a natural language question over indexed documents."""
    service = AskService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        llm_client=get_llm_client(),
    )
    return await service.ask(request)
