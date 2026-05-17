# Document Upload RAG Prototype

Small FastAPI project for learning a local-first RAG pipeline with:

- document upload
- text extraction from `.txt` files
- chunking with `langchain-text-splitters`
- embeddings with `sentence-transformers`
- vector search with local Qdrant
- answer generation with a local `llama-server` model endpoint

## Requirements

- Python 3.11+
- virtual environment recommended

## Install

```bash
pip install -r requirements.txt
```

## Run

Start the API with `uvicorn`:

```bash
uvicorn main:app --reload
```

The app will be available at:

- `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## Local Model

`/ask` expects a local OpenAI-compatible model endpoint to be running.

Current setup:

```bash
~/llama.cpp/build/bin/llama-server -m Qwen_Qwen3-14B-Q4_K_M.gguf --port 8080
```

Current `GenerationService` target:

- `http://127.0.0.1:8080/v1/chat/completions`

## Current Flow

This project currently supports `.txt` uploads only.

1. Upload a document
2. Extract text
3. Chunk the text
4. Embed the chunks into Qdrant
5. Run semantic search across all indexed documents
6. Ask a question against the indexed corpus

There are two upload flows available:

- `POST /upload_v1`
  - upload only
- `POST /upload_v2`
  - upload, extract, chunk, and embed in a single request

## API Endpoints

- `GET /health`
- `POST /upload_v1`
  - upload a `.txt` document only
- `POST /upload_v2`
  - upload and run the full indexing pipeline
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
  - retrieve relevant chunks and generate a human-readable answer from the indexed corpus

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

Run semantic search across all indexed documents:

```bash
curl -X POST "http://127.0.0.1:8000/semantic-search" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is retrieval augmented generation","limit":3}'
```

Ask a question across all indexed documents:

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"what is retrieval augmented generation","limit":3}'
```

## Sample Fixtures

Sample `.txt` files for manual testing live under:

```text
tests/fixtures/
```

Current fixtures:

- `python_rag_intro.txt`
- `vector_database_notes.txt`
- `laravel_queues.txt`

## Notes

- Document metadata, extracted text, and chunks are persisted in SQLite.
- Uploaded file contents are stored on disk in `uploads/`.
- Vector data is stored in local Qdrant data on disk.
- Semantic search and `/ask` only work after at least one document has been embedded.
- `/ask` uses retrieved chunks as context for a local OpenAI-compatible `llama-server` endpoint.
