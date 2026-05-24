from sqlalchemy import text

from app.db.database import SessionLocal
from app.db.models import DocumentChunk

FTS_TABLE_NAME = "document_chunks_fts"


class LexicalSearchService:
    def __init__(self) -> None:
        self._ensure_table()

    def _ensure_table(self) -> None:
        with SessionLocal() as session:
            session.execute(text(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE_NAME}
                USING fts5(
                    document_chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    chunk_index UNINDEXED,
                    original_filename UNINDEXED,
                    text
                )
            """))
            session.commit()

    def _normalize_query(self, query: str) -> str:
        # TODO: need to further normalize to strip unsafe character
        tokens = query.lower().strip().split()
        return " ".join(tokens)

    def delete_document_chunks(self, document_id: str) -> None:
        with SessionLocal() as session:
            session.execute(
                text(f"""
                DELETE FROM {FTS_TABLE_NAME}
                WHERE document_id = :document_id;
            """),
                {"document_id": document_id},
            )
            session.commit()

    def index_document_chunks(
        self, document_id: str, original_filename: str, chunks: list[DocumentChunk]
    ) -> None:
        self.delete_document_chunks(document_id)

        if len(chunks) == 0:
            return

        rows = [
            {
                "document_chunk_id": chunk.id,
                "document_id": document_id,
                "chunk_index": chunk.chunk_index,
                "original_filename": original_filename,
                "text": chunk.text,
            }
            for chunk in chunks
        ]

        with SessionLocal() as session:
            session.execute(
                text(f"""
                    INSERT INTO {FTS_TABLE_NAME} (
                        document_chunk_id,
                        document_id,
                        chunk_index,
                        original_filename,
                        text
                    )
                    VALUES (
                        :document_chunk_id,
                        :document_id,
                        :chunk_index,
                        :original_filename,
                        :text
                    )
                """),
                rows,
            )
            session.commit()

    def search(self, query: str, limit: int) -> list[dict]:
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        with SessionLocal() as session:
            result = session.execute(
                text(f"""
                SELECT
                    document_id,
                    original_filename,
                    chunk_index,
                    text,
                    bm25({FTS_TABLE_NAME}) as score
                FROM {FTS_TABLE_NAME}
                WHERE {FTS_TABLE_NAME} MATCH :query
                ORDER BY bm25({FTS_TABLE_NAME})
                LIMIT :limit
            """),
                {"query": normalized_query, "limit": limit},
            )

            rows = result.mappings().all()

        return [
            {
                "document_id": row["document_id"],
                "original_filename": row["original_filename"],
                "chunk_index": row["chunk_index"],
                "score": row["score"],
                "text": row["text"],
            }
            for row in rows
        ]
