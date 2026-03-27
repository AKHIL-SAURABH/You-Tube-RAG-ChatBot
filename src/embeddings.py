from sentence_transformers import SentenceTransformer
import torch


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"⚡ Embedding model loading on: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 🔥 improves retrieval quality
        )

        # ✅ Convert to Python lists (important for Chroma)
        embeddings = [emb.tolist() for emb in embeddings]

        print(f"📐 Generated embeddings for {len(texts)} texts")

        return embeddings