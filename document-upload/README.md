# Document Upload RAG Prototype

Small FastAPI project for learning a local-first RAG pipeline with:

- document upload
- text extraction from `.txt` and `.pdf` files
- chunking with `langchain-text-splitters`
- embeddings with `sentence-transformers`
- vector search with local Qdrant
- answer generation with a local `llama-server` model endpoint
- a small mobile-first chat UI backed by WebSocket streaming

## Requirements

- Python 3.11+
- virtual environment recommended

## Install

```bash
pip install -r requirements.txt
```

## Run

Start the app with `uvicorn`:

```bash
uvicorn asgi:app --reload --host 0.0.0.0
```

The app will be available at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Chat UI: `http://127.0.0.1:8000/chat`

## Local Model

Both `POST /ask` and `WS /ws/chat` expect a local OpenAI-compatible model endpoint to be running.

Current setup:

```bash
~/llama.cpp/build/bin/llama-server -m Qwen_Qwen3-14B-Q4_K_M.gguf --port 8080
```

Current `GenerationService` target:

- `http://127.0.0.1:8080/v1/chat/completions`

## Current Flow

1. Upload a document
2. Extract text
3. Chunk the text
4. Embed the chunks into Qdrant
5. Run semantic search across all indexed documents
6. Ask a question against the indexed corpus
7. Stream an answer in the browser over WebSocket
8. Optionally upload and index in the background with job polling

There are three upload flows available:

- `POST /upload_v1`
  - upload only
- `POST /upload_v2`
  - upload, extract, chunk, and embed in a single request
- `POST /upload_async`
  - upload a document, create a background indexing job, and return a queued job response immediately

Supported upload types right now:

- `.txt`
- `.pdf`

## API Endpoints

- `GET /health`
- `GET /chat`
  - serve the chat UI
- `POST /upload_v1`
  - upload a document only
- `POST /upload_v2`
  - upload and run the full indexing pipeline
- `POST /upload_async`
  - upload and enqueue the indexing pipeline as a background job
- `GET /{document_id}`
  - fetch persisted document metadata and chunks
- `POST /{document_id}/extract`
- `POST /{document_id}/chunk`
- `POST /{document_id}/search`
  - document-scoped keyword search
- `POST /{document_id}/embed`
  - embed and index one document into Qdrant
- `POST /semantic-search`
  - corpus-wide semantic search across all embedded documents
- `POST /ask`
  - retrieve relevant chunks and generate one non-streaming answer from the indexed corpus
- `GET /jobs/{job_id}`
  - fetch the current status of a background indexing job
- `WS /ws/chat`
  - retrieve relevant chunks and stream answer tokens to the chat UI

## Background Job Flow

`POST /upload_async` stores the uploaded document, creates a job with status `queued`, and schedules the indexing pipeline in a FastAPI background task.

Initial response example:

```json
{
  "job_id": "job-123",
  "job_type": "document_index",
  "document_id": "doc-123",
  "status": "queued",
  "error_message": null,
  "created_at": "2026-05-23T12:00:00",
  "started_at": null,
  "finished_at": null
}
```

Poll the job with:

```text
GET /jobs/{job_id}
```

Current job statuses:

- `queued`
- `running`
- `completed`
- `failed`

## Chat Flow

The browser chat page at `GET /chat` opens a websocket connection to `WS /ws/chat`.

Client messages:

```json
{"query":"what is retrieval augmented generation","limit":3}
```

Server events:

```json
{"type":"status","message":"retrieving context"}
{"type":"status","message":"generating answer"}
{"type":"token","value":"Retrieval "}
{"type":"done","answer":"Retrieval augmented generation...","sources":[...]}
{"type":"error","message":"Generation service is unavailable"}
```

`POST /ask` is still available for the non-streaming request/response flow and is useful for Swagger testing.

## Example Manual Workflow

### One-step upload and indexing

Upload, extract, chunk, and embed in one request:

```bash
curl -X POST "http://127.0.0.1:8000/upload_v2" \
  -F "file=@tests/fixtures/python_rag_intro.txt;type=text/plain"
```

### Step-by-step upload flow

Upload a file only:

```bash
curl -X POST "http://127.0.0.1:8000/upload_v1" \
  -F "file=@tests/fixtures/python_rag_intro.txt;type=text/plain"
```

Then use the returned `document_id` in the next steps:

```bash
curl -X POST "http://127.0.0.1:8000/<document_id>/extract"
curl -X POST "http://127.0.0.1:8000/<document_id>/chunk"
curl -X POST "http://127.0.0.1:8000/<document_id>/embed"
```

### Ask a question over HTTP

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is retrieval augmented generation","limit":3}'
```

### Upload and index in the background

```bash
curl -X POST "http://127.0.0.1:8000/upload_async" \
  -F "file=@tests/fixtures/python_rag_intro.txt;type=text/plain"
```

Then poll the returned `job_id`:

```bash
curl "http://127.0.0.1:8000/jobs/<job_id>"
```

### Open the chat UI

Visit:

```text
http://127.0.0.1:8000/chat
```

The UI will connect to `WS /ws/chat` automatically and stream the answer into the assistant bubble.

## Tests

Run the test suite with:

```bash
pytest
```

Current coverage includes:

- upload pipeline behavior
- async upload job creation and polling behavior
- `/ask` request/response behavior
- websocket happy path
- websocket no-context path
- websocket generation-error path


## TODOs:
4. hybrid retrieval + reranking (1 session)
5. structured citation output (1/2 session)
6. logging + latency metrics (1/2 session)
7. Docker Compose (1/2 session)
8. RAG eval tests (1 session)
9. breakdown main.py more (ask.py, health.py...) (1/2 session)
