from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from config import CHROMA_PATH


class CustomEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        embeddings = self.model.embed(texts)
        return [list(e) for e in embeddings]  # ✅ ensure correct format

    def embed_query(self, text):
        embedding = self.model.embed([text])[0]
        return list(embedding)  # ✅ ensure correct format


def create_vectorstore(chunks, embedding_model):
    print("📦 Creating ChromaDB...")

    embedding = CustomEmbedding(embedding_model)

    if len(chunks) == 0:
        raise ValueError("❌ No chunks provided to vectorstore!")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ Stored {len(chunks)} chunks in ChromaDB")

    db.persist()

    return db


def load_vectorstore(embedding_model):
    print("📂 Loading existing ChromaDB...")

    embedding = CustomEmbedding(embedding_model)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    print("✅ ChromaDB loaded")

    return db