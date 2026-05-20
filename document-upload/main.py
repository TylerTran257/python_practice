from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.routing import APIRouter

from database import Base, engine
from document_service import DocumentData, DocumentService
from embedding_service import EmbeddingService
from generation_service import GenerationService, GenerationServiceError
from schemas import AskRequest, SearchRequest
from text_extractor import TextExtractor
from vector_store_service import VectorStoreService

router = APIRouter()


def create_app(document_service=None, generation_service=None) -> FastAPI:
    app = FastAPI()
    Base.metadata.create_all(bind=engine)

    resolved_document_service = document_service
    if document_service is None:
        embedding_service = EmbeddingService()
        vector_store_service = VectorStoreService()
        text_extractor = TextExtractor()
        resolved_document_service = DocumentService(
            embedding_service, vector_store_service, text_extractor
        )

    resolved_generation_service = generation_service
    if generation_service is None:
        resolved_generation_service = GenerationService()
    app.state.document_service = resolved_document_service
    app.state.generation_service = resolved_generation_service

    app.include_router(router)
    return app


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload_v1")
async def upload_document_v1(
    request: Request, file: UploadFile = File(...)
) -> DocumentData:
    return await request.app.state.document_service.create_document(file)


@router.get("/{document_id}")
def get_document(request: Request, document_id: str) -> DocumentData:
    return request.app.state.document_service.get_document(document_id)


@router.post("/{document_id}/extract")
def extract_document(request: Request, document_id: str) -> DocumentData:
    return request.app.state.document_service.extract_text(document_id)


@router.post("/{document_id}/chunk")
def chunk_document(request: Request, document_id: str) -> dict[str, str | int]:
    return request.app.state.document_service.chunk_document(document_id)


@router.post("/{document_id}/search")
def search_document(
    request: Request, document_id: str, searchRequest: SearchRequest
) -> dict:
    return request.app.state.document_service.search_document(
        document_id, searchRequest.query, searchRequest.limit
    )


@router.post("/{document_id}/embed")
def embed_document(request: Request, document_id: str) -> dict:
    return request.app.state.document_service.embed_document(document_id)


@router.post("/upload_v2")
async def upload_document_v2(
    request: Request, file: UploadFile = File(...)
) -> DocumentData:
    uploaded_file = await request.app.state.document_service.create_document(file)
    document_id = uploaded_file.get("document_id")
    if document_id is None:
        raise HTTPException(status_code=500, detail="Document Id Not Found")

    request.app.state.document_service.extract_text(document_id)
    request.app.state.document_service.chunk_document(document_id)
    request.app.state.document_service.embed_document(document_id)

    return request.app.state.document_service.get_document(document_id)


@router.post("/semantic-search")
def semantic_search_document(request: Request, searchRequest: SearchRequest) -> dict:
    return request.app.state.document_service.semantic_search(
        searchRequest.query, searchRequest.limit
    )


@router.post("/ask")
def ask(request: Request, askRequest: AskRequest) -> dict:
    contexts = request.app.state.document_service.retrieve_context(
        askRequest.query, askRequest.limit
    )
    if len(contexts) == 0:
        return {
            "query": askRequest.query,
            "answer": "",
            "match_count": 0,
            "sources": [],
        }

    try:
        answer = request.app.state.generation_service.answer_question(
            askRequest.query, contexts
        )
    except GenerationServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return {
        "query": askRequest.query,
        "answer": answer if len(answer) != 0 else "",
        "match_count": len(contexts),
        "sources": contexts,
    }
