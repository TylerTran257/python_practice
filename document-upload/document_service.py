from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from fastapi import HTTPException, UploadFile

UPLOAD_DIR = Path("uploads")
MAX_FILE_SIZE = 1024 * 1024  # 1 MB
ALLOWED_CONTENT_TYPES = {"text/plain"}


class DocumentData(TypedDict, total=False):
    document_id: str
    original_filename: str
    stored_filename: str
    status: str
    size_bytes: int
    extracted_text: str
    chunks: list[str]


class DocumentService:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentData] = dict()

    async def create_document(self, file: UploadFile) -> DocumentData:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Documentname is required")

        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400, detail="Only .txt files are supported right now"
            )

        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, detail="Document is too large. Max size is 1 MB"
            )
        UPLOAD_DIR.mkdir(exist_ok=True)

        document_id = str(uuid4())
        saved_filename = f"{document_id}.txt"
        saved_path = UPLOAD_DIR / saved_filename
        saved_path.write_bytes(contents)

        document: DocumentData = {
            "document_id": document_id,
            "original_filename": file.filename or "unknown",
            "stored_filename": saved_filename,
            "status": "uploaded",
            "size_bytes": len(contents),
        }
        self.documents[document_id] = document
        return document

    def get_document(self, document_id: str) -> DocumentData:
        document = self.documents.get(document_id)

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return self.documents[document_id]

    def extract_text(self, document_id: str) -> DocumentData:
        document = self.documents.get(document_id)

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("status") == "text_extracted":
            return document

        if document.get("status") != "uploaded":
            raise HTTPException(status_code=400, detail="Document state error")

        stored_filename = document.get("stored_filename")
        saved_path = UPLOAD_DIR / str(stored_filename)
        content = saved_path.read_bytes()

        document["status"] = "text_extracted"
        document["extracted_text"] = content.decode("utf-8")
        return document

    def chunk_document(self, document_id: str) -> dict[str, str | int]:
        document = self.documents[document_id]

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("status") == "chunked":
            chunks = document.get("chunks")
            return {
                "document_id": document_id,
                "status": "chunked",
                "chunk_count": len(chunks or []),
            }

        if document.get("status") != "text_extracted":
            raise HTTPException(
                status_code=409,
                detail="Document must be text_extracted before chunking",
            )

        extracted_text = document.get("extracted_text")
        if extracted_text is None:
            raise HTTPException(
                status_code=409,
                detail="Document must be text_extracted before chunking",
            )
        chunk_size = 500

        chunks = []
        for start in range(0, len(extracted_text), chunk_size):
            chunk = extracted_text[start : start + chunk_size]
            chunks.append(chunk)

        document["chunks"] = chunks
        document["status"] = "chunked"

        return {
            "document_id": document_id,
            "status": "chunked",
            "chunk_count": len(chunks),
        }

    def search_document(
        self, document_id: str, query: str, limit: int
    ) -> dict[str, str | int | list[str]]:
        document = self.documents.get(document_id)

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.get("status") != "chunked":
            raise HTTPException(
                status_code=409,
                detail="Document must be chunked before searching",
            )

        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query must not be empty",
            )

        if limit <= 0:
            raise HTTPException(
                status_code=400,
                detail="Limit must be greater than 0",
            )

        chunks = document.get("chunks")
        if chunks is None:
            raise HTTPException(
                status_code=500,
                detail="Chunks missing",
            )

        normalized_query = query.lower()
        results = []

        for index, chunk in enumerate(chunks):
            score = chunk.lower().count(normalized_query)

            if score > 0:
                results.append(
                    {
                        "chunk_index": index,
                        "score": score,
                        "text": chunk,
                    }
                )
        results.sort(key=lambda item: item["score"], reverse=True)
        top_results = results[:limit]

        return {
            "document_id": document_id,
            "query": query,
            "match_count": len(top_results),
            "results": top_results,
        }
