from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict

from app.db.database import SessionLocal
from app.db.models import Document, DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService
from app.settings import settings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

ALLOWED_CONTENT_TYPES = {"text/plain", "application/pdf"}


class DocumentData(TypedDict, total=False):
    document_id: str
    original_filename: str
    stored_filename: str
    status: str
    size_bytes: int
    extracted_text: str | None
    chunks: list[str]
    chunk_embeddings: list[list[float]]


class DocumentService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
        text_extractor: TextExtractor,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.text_extractor = text_extractor

    async def create_document(self, file: UploadFile) -> DocumentData:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Documentname is required")

        if file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400, detail=f"{file.content_type} is not allowed"
            )

        contents = await file.read()

        if len(contents) > settings.max_file_size:
            raise HTTPException(
                status_code=400, detail="Document is too large. Max size is 1 MB"
            )
        settings.upload_dir.mkdir(exist_ok=True)

        document_id = str(uuid4())
        original_suffix = Path(file.filename).suffix.lower()
        saved_filename = f"{document_id}{original_suffix}"
        saved_path = settings.upload_dir / saved_filename
        saved_path.write_bytes(contents)

        with SessionLocal() as session:
            document = Document(
                id=document_id,
                original_filename=file.filename or "unknown",
                stored_filename=saved_filename,
                status="uploaded",
                size_bytes=len(contents),
                extracted_text=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return self.serialize_document(document)

    def get_document(self, document_id: str) -> DocumentData:
        with SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")
            return self.serialize_document(document)

    def search_document(
        self, document_id: str, query: str, limit: int
    ) -> dict[str, str | int | list[str]]:
        with SessionLocal() as session:
            document = session.get(Document, document_id)

            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

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

            chunks = [chunk.text for chunk in document.chunks]
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

    def extract_text(self, document_id: str) -> DocumentData:
        with SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.status == "text_extracted":
                return self.serialize_document(document)

            if document.status != "uploaded":
                raise HTTPException(status_code=400, detail="Document state error")

            stored_filename = document.stored_filename
            saved_path = settings.upload_dir / str(stored_filename)

            document.extracted_text = self.text_extractor.extract(saved_path)
            document.status = "text_extracted"
            document.updated_at = datetime.now()

            session.commit()
            session.refresh(document)
            return self.serialize_document(document)

    def chunk_document(self, document_id: str) -> dict[str, str | int]:
        with SessionLocal() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.status == "chunked":
                chunks = document.chunks
                return {
                    "document_id": document_id,
                    "status": "chunked",
                    "chunk_count": len(chunks or []),
                }

            if document.status != "text_extracted":
                raise HTTPException(
                    status_code=409,
                    detail="Document must be text_extracted before chunking",
                )

            extracted_text = document.extracted_text
            if extracted_text is None:
                raise HTTPException(
                    status_code=409,
                    detail="Document must be text_extracted before chunking",
                )

            document.chunks.clear()

            chunks = text_splitter.split_text(extracted_text)
            for index, chunk_text in enumerate(chunks):
                document.chunks.append(
                    DocumentChunk(
                        chunk_index=index,
                        text=chunk_text,
                        created_at=datetime.now(),
                    )
                )

            document.status = "chunked"
            document.updated_at = datetime.now()
            session.commit()

            return {
                "document_id": document_id,
                "status": "chunked",
                "chunk_count": len(chunks),
            }

    def embed_document(self, document_id: str) -> dict[str, str | int | list[str]]:
        with SessionLocal() as session:
            document = session.get(Document, document_id)

            if document is None:
                raise HTTPException(status_code=404, detail="Document not found")

            if document.status == "embedded":
                return {
                    "document_id": document.id or "",
                    "status": document.status or "",
                    "embedding_count": len(document.chunks),
                }

            if document.status != "chunked":
                raise HTTPException(
                    status_code=409,
                    detail="Document must be chunked before embedding",
                )

            chunks = [chunk.text for chunk in document.chunks]
            chunk_embeddings = self.embedding_service.embed_texts(chunks)

            self.vector_store_service.upsert_document_chunks(
                document_id,
                document.original_filename,
                chunks,
                chunk_embeddings,
            )

            document.status = "embedded"
            document.updated_at = datetime.now()
            session.commit()

            return {
                "document_id": document.id,
                "status": "embedded",
                "embedding_count": len(chunk_embeddings),
            }

    def retrieve_context(self, query: str, limit: int) -> list[dict]:
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

        if not self.vector_store_service.has_indexed_chunks():
            raise HTTPException(
                status_code=409,
                detail="At least one document must be embedded before semantic searching",
            )

        query_embedding = self.embedding_service.embed_text(query)

        return self.vector_store_service.search(query_embedding, limit)

    def semantic_search(
        self, query: str, limit: int
    ) -> dict[str, str | int | list[str] | list[dict[str, str | int]]]:
        results = self.retrieve_context(query, limit)

        return {"query": query, "match_count": len(results), "results": results}

    def serialize_document(self, document: Document) -> DocumentData:
        chunks = [chunk.text for chunk in document.chunks]

        return {
            "document_id": document.id,
            "original_filename": document.original_filename,
            "stored_filename": document.stored_filename,
            "status": document.status,
            "size_bytes": document.size_bytes,
            "extracted_text": document.extracted_text,
            "chunks": chunks,
        }
