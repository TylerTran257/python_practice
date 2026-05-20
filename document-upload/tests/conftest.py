import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from main import create_app


class FakeDocumentService:
    def __init__(self) -> None:
        self.calls = []
        self.contexts = []

    async def create_document(self, file):
        self.calls.append(("create_document", file.filename))
        return {"document_id": "doc-123"}

    def extract_text(self, document_id):
        self.calls.append(("extract_text", document_id))
        return {"document_id": document_id, "status": "text_extracted"}

    def chunk_document(self, document_id):
        self.calls.append(("chunk_document", document_id))
        return {"document_id": document_id, "status": "chunked", "chunk_count": 2}

    def embed_document(self, document_id):
        self.calls.append(("embed_document", document_id))
        return {"document_id": document_id, "status": "embedded", "embedding_count": 2}

    def get_document(self, document_id):
        self.calls.append(("get_document", document_id))
        return {
            "document_id": document_id,
            "original_filename": "python_rag_intro.txt",
            "stored_filename": "doc-123.txt",
            "status": "embedded",
            "size_bytes": 123,
            "extracted_text": "some extracted text",
            "chunks": ["chunk one", "chunk two"],
        }

    def retrieve_context(self, query, limit):
        self.calls.append(("retrieve_context", query, limit))
        return self.contexts


class FakeGenerationService:
    def __init__(self, answer="", error=None) -> None:
        self.answer = answer
        self.error = error
        self.calls = []

    def answer_question(self, question, sources):
        self.calls.append(("answer_question", question, sources))

        if self.error is not None:
            raise self.error

        return self.answer


@pytest.fixture
def fake_document_service():
    return FakeDocumentService()


@pytest.fixture
def fake_generation_service():
    return FakeGenerationService()


@pytest.fixture
def client(fake_document_service, fake_generation_service):
    app = create_app(
        document_service=fake_document_service,
        generation_service=fake_generation_service,
    )
    return TestClient(app)
