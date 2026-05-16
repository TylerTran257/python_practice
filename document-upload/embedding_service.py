from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self) -> None:
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_text(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
