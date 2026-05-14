import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    dot_product = 0.0
    for left, right in zip(a, b):
        dot_product += left * right

    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class EmbeddingService:
    def __init__(self) -> None:
        return None

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * 8
        normalized_text = text.lower()

        for char in normalized_text:
            bucket_index = ord(char) % 8
            vector[bucket_index] += 1.0

        return vector
