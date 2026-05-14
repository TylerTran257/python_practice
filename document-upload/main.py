from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from document_service import DocumentData, DocumentService

app = FastAPI()

document_service = DocumentService()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=3, gt=0, le=10)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> DocumentData:
    return await document_service.create_document(file)


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> DocumentData:
    return document_service.get_document(document_id)


@app.post("/documents/{document_id}/extract")
def extract_document(document_id: str) -> DocumentData:
    return document_service.extract_text(document_id)


@app.post("/documents/{document_id}/chunk")
def chunk_document(document_id: str) -> dict[str, str | int]:
    return document_service.chunk_document(document_id)


@app.post("/documents/{document_id}/search")
def search_document(document_id: str, request: SearchRequest) -> dict:
    return document_service.search_document(document_id, request.query, request.limit)
