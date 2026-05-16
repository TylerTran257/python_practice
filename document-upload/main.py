from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from document_service import DocumentData, DocumentService
from embedding_service import EmbeddingService
from VectorStoreService import VectorStoreService

app = FastAPI()

embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()
document_service = DocumentService(embedding_service, vector_store_service)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/")
async def upload_document(file: UploadFile = File(...)) -> DocumentData:
    return await document_service.create_document(file)


@app.get("/{document_id}")
def get_document(document_id: str) -> DocumentData:
    return document_service.get_document(document_id)


@app.post("/{document_id}/extract")
def extract_document(document_id: str) -> DocumentData:
    return document_service.extract_text(document_id)


@app.post("/{document_id}/chunk")
def chunk_document(document_id: str) -> dict[str, str | int]:
    return document_service.chunk_document(document_id)


@app.post("/{document_id}/search")
def search_document(document_id: str, request: SearchRequest) -> dict:
    return document_service.search_document(document_id, request.query, request.limit)


@app.post("/{document_id}/embed")
def embed_document(document_id: str) -> dict:
    return document_service.embed_document(document_id)


@app.post("/semantic-search")
def semantic_search_document(request: SearchRequest) -> dict:
    return document_service.semantic_search(request.query, request.limit)
