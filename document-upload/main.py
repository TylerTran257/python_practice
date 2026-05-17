from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from database import Base, engine
from document_service import DocumentData, DocumentService
from embedding_service import EmbeddingService
from vector_store_service import VectorStoreService

app = FastAPI()
Base.metadata.create_all(bind=engine)

embedding_service = EmbeddingService()
vector_store_service = VectorStoreService()
document_service = DocumentService(embedding_service, vector_store_service)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload_v1")
async def upload_document_v1(file: UploadFile = File(...)) -> DocumentData:
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


@app.post("/upload_v2")
async def upload_document_v2(file: UploadFile = File(...)) -> DocumentData:
    uploaded_file = await document_service.create_document(file)
    document_id = uploaded_file.get("document_id")
    if document_id is None:
        raise HTTPException(status_code=500, detail="Document Id Not Found")

    document_service.extract_text(document_id)
    document_service.chunk_document(document_id)
    document_service.embed_document(document_id)

    return document_service.get_document(document_id)


@app.post("/semantic-search")
def semantic_search_document(request: SearchRequest) -> dict:
    return document_service.semantic_search(request.query, request.limit)
