from src.youtube_loader import load_youtube_video
from src.splitter import split_documents
from src.embeddings import EmbeddingModel
from src.vectorstore import create_vectorstore
from src.llm import LLMModel


class RAGPipeline:
    def __init__(self, video_url):
        self.embedding_model = EmbeddingModel()
        self.llm = LLMModel()

        print("⚠️ Rebuilding database from YouTube video...")

        docs = load_youtube_video(video_url)

        if not docs:
            raise ValueError("❌ No transcript found for this video.")

        print(f"📄 Loaded documents: {len(docs)}")

        chunks = split_documents(docs)

        print(f"✂️ Total chunks created: {len(chunks)}")

        if len(chunks) == 0:
            raise ValueError("❌ Chunking failed — no chunks created.")

        self.db = create_vectorstore(chunks, self.embedding_model)

        print("✅ ChromaDB created successfully!")

    def query(self, question):
        # 🔥 Improve query (smart boosting)
        q = question.lower()
        if "who" in q:
            question += " instructor teacher name"
        elif "what" in q:
            question += " explanation overview"
        elif "how" in q:
            question += " steps process"

        # 🔍 Retrieve with scores (better than retriever.invoke)
        docs_with_scores = self.db.similarity_search_with_score(question, k=5)

        # Sort by best similarity (lower score = better)
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

        # Take top 3 best chunks
        docs = [doc for doc, score in docs_with_scores[:3]]

        print(f"\n📚 Retrieved docs count: {len(docs)}")

        if not docs:
            return "❌ No relevant context found."

        # 🔥 Add intro chunk (important for general questions)
        try:
            intro_chunk = self.db.similarity_search("SQL tutorial introduction overview", k=1)[0]
            docs = [intro_chunk] + docs
        except:
            pass

        # 🧠 Build limited context (avoid overflow)
        MAX_CONTEXT_CHARS = 2000

        context = ""
        for doc in docs:
            if len(context) + len(doc.page_content) < MAX_CONTEXT_CHARS:
                context += doc.page_content + "\n\n"
            else:
                break

        # 🔍 DEBUG
        print("\n🔍 Retrieved Context (first 1000 chars):\n")
        print(context[:1000])
        print("\n" + "=" * 60)

        # 🔥 Improved prompt (forces summarization)
        prompt = f"""
You are an expert assistant.

Read the context and answer the question clearly.

Rules:
- Answer in 2-3 sentences
- Do NOT copy large chunks
- Summarize in your own words
- Be precise and relevant
- If answer is not found, say:
  "I could not find the answer in the video."

Context:
{context}

Question: {question}

Final Answer:
"""

        response = self.llm.generate(prompt)

        return response.strip()