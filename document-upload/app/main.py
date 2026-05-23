from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.routing import APIRouter
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from app.api.schemas import AskRequest, SearchRequest
from app.db.database import Base, engine
from app.services.document_service import DocumentData, DocumentService
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService, GenerationServiceError
from app.services.text_extractor import TextExtractor
from app.services.vector_store_service import VectorStoreService

router = APIRouter()
templates = Jinja2Templates(directory="templates")


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


@router.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {})


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


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            payload = await websocket.receive_json()

            try:
                ask_request = AskRequest.model_validate(payload)
            except ValidationError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid chat payload"}
                )
                continue

            await websocket.send_json(
                {"type": "status", "message": "retrieving context"}
            )

            try:
                contexts = websocket.app.state.document_service.retrieve_context(
                    ask_request.query,
                    ask_request.limit,
                )
            except HTTPException as exc:
                message = (
                    exc.detail
                    if isinstance(exc.detail, str)
                    else "Failed to retrieve document context"
                )
                await websocket.send_json({"type": "error", "message": message})
                continue
            except Exception:
                await websocket.send_json(
                    {"type": "error", "message": "Failed to retrieve document context"}
                )
                continue

            if not contexts:
                await websocket.send_json({"type": "done", "answer": "", "sources": []})
                continue
            await websocket.send_json(
                {
                    "type": "status",
                    "message": "generating answer",
                }
            )

            full_answer = ""
            try:
                async for (
                    token
                ) in websocket.app.state.generation_service.stream_answer_question(
                    ask_request.query,
                    contexts,
                ):
                    full_answer += token
                    await websocket.send_json({"type": "token", "value": token})
            except GenerationServiceError as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            await websocket.send_json(
                {"type": "done", "answer": full_answer, "sources": contexts}
            )
    except WebSocketDisconnect:
        pass
