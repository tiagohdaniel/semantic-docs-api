from fastapi import APIRouter, HTTPException

from app.schemas.models import SourcesResponse, SourceItem
from app.dependencies import get_vector_store

router = APIRouter()


@router.get("/sources", response_model=SourcesResponse, summary="List indexed sources")
def list_sources():
    """List all indexed sources with their chunk counts."""
    vector_store = get_vector_store()
    raw = vector_store.list_sources()
    sources = [SourceItem(**s) for s in raw]
    return SourcesResponse(sources=sources, total=len(sources))


@router.delete("/sources/{source_id}", summary="Delete an indexed source")
def delete_source(source_id: str):
    """Remove all chunks for a source from the index."""
    vector_store = get_vector_store()
    deleted = vector_store.delete_source(source_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")
    return {"deleted_chunks": deleted, "source_id": source_id}
