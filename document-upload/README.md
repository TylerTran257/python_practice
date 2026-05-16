# Document Upload RAG Prototype

Small FastAPI project for learning a local-first RAG pipeline with:

- document upload
- text extraction from `.txt` files
- chunking with `langchain-text-splitters`
- embeddings with `sentence-transformers`
- vector search with local Qdrant

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

## Current Flow

This project currently supports `.txt` uploads only.

1. Upload a document
2. Extract text
3. Chunk the text
4. Embed the chunks
5. Run semantic search across all indexed documents

## API Endpoints

- `GET /health`
- `POST /`
  - upload a `.txt` document
- `GET /{document_id}`
  - fetch document metadata from in-memory app state
- `POST /{document_id}/extract`
- `POST /{document_id}/chunk`
- `POST /{document_id}/search`
  - document-scoped keyword search
- `POST /{document_id}/embed`
  - embed and index one document into Qdrant
- `POST /semantic-search`
  - corpus-wide semantic search across all embedded documents

## Example Manual Workflow

Upload a file:

```bash
curl -X POST "http://127.0.0.1:8000/" \
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

- Document metadata is currently stored in memory.
- Vector data is stored in local Qdrant data on disk.
- Semantic search only works after at least one document has been embedded.
- This is still a retrieval-focused prototype; answer generation is not added yet.
