import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
from generation_service import GenerationServiceError

CONTEXTS = [
    {
        "document_id": "doc-1",
        "original_filename": "python_rag_intro.txt",
        "chunk_index": 0,
        "score": 0.91,
        "text": "Retrieval augmented generation combines retrieval with generation.",
    }
]


class FakeDocumentService:
    def __init__(self, contexts) -> None:
        self.contexts = contexts
        self.calls = []

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


def test_ask_returns_answer_and_sources(monkeypatch):
    import main

    fake_document_service = FakeDocumentService(CONTEXTS)
    fake_generation_service = FakeGenerationService(
        "Retrieval augmented generation combines retrieval with generation.",
    )
    monkeypatch.setattr(main, "document_service", fake_document_service)
    monkeypatch.setattr(main, "generation_service", fake_generation_service)

    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"query": "what is retrieval augmented generation", "limit": 3},
    )

    assert response.status_code == 200

    assert response.json() == {
        "query": "what is retrieval augmented generation",
        "answer": "Retrieval augmented generation combines retrieval with generation.",
        "match_count": 1,
        "sources": CONTEXTS,
    }


def test_ask_returns_502_when_generation_fails(monkeypatch):
    import main

    fake_document_service = FakeDocumentService(CONTEXTS)
    fake_generation_service = FakeGenerationService(
        error=GenerationServiceError("Some errors")
    )
    monkeypatch.setattr(main, "document_service", fake_document_service)
    monkeypatch.setattr(main, "generation_service", fake_generation_service)

    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"query": "what is retrieval augmented generation", "limit": 3},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Some errors"}


def test_ask_returns_empty_answer_when_no_context_found(monkeypatch):
    import main

    fake_document_service = FakeDocumentService([])
    fake_generation_service = FakeGenerationService(
        error=AssertionError("Generation should not be called")
    )
    monkeypatch.setattr(main, "document_service", fake_document_service)
    monkeypatch.setattr(main, "generation_service", fake_generation_service)

    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={"query": "what is retrieval augmented generation", "limit": 3},
    )

    assert response.status_code == 200

    assert response.json() == {
        "query": "what is retrieval augmented generation",
        "answer": "",
        "match_count": 0,
        "sources": [],
    }
    assert fake_generation_service.calls == []
