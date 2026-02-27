from fastapi import APIRouter, Depends

from app.schemas.models import AskRequest, AskResponse
from app.dependencies import get_embedding_service, get_vector_store, get_llm_client
from app.services.ask_service import AskService

router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Query indexed documents")
async def ask_question(
    request: AskRequest,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    llm_client=Depends(get_llm_client),
):
    """Ask a natural language question over indexed documents.

    Returns an LLM-generated answer grounded on retrieved chunks,
    with source references and relevance scores.
    """
    service = AskService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_client=llm_client,
    )
    return await service.ask(request)
