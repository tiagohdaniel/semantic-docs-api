from fastapi import APIRouter

from app.schemas.models import IndexRequest, IndexResponse
from app.dependencies import get_embedding_service, get_vector_store, get_chunker
from app.services.index_service import IndexService

router = APIRouter()


@router.post("/index", response_model=IndexResponse, summary="Index a document")
def index_document(request: IndexRequest):
    """Index a document for semantic search. Re-indexing replaces existing chunks."""
    service = IndexService(
        chunker=get_chunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )
    return service.index(request)
