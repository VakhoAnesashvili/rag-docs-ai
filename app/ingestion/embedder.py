import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            QUERY_PREFIX + query, normalize_embeddings=True, convert_to_numpy=True
        )