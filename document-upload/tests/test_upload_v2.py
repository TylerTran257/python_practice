import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))


class FakeDocumentService:
    def __init__(self) -> None:
        self.calls = []

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


def test_upload_v2_runs_full_pipeline_and_returns_final_document(monkeypatch):
    import main

    fake_service = FakeDocumentService()
    monkeypatch.setattr(main, "document_service", fake_service)

    client = TestClient(main.app)

    response = client.post(
        "/upload_v2",
        files={"file": ("python_rag_intro.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200

    assert response.json() == {
        "document_id": "doc-123",
        "original_filename": "python_rag_intro.txt",
        "stored_filename": "doc-123.txt",
        "status": "embedded",
        "size_bytes": 123,
        "extracted_text": "some extracted text",
        "chunks": ["chunk one", "chunk two"],
    }

    assert fake_service.calls == [
        ("create_document", "python_rag_intro.txt"),
        ("extract_text", "doc-123"),
        ("chunk_document", "doc-123"),
        ("embed_document", "doc-123"),
        ("get_document", "doc-123"),
    ]
