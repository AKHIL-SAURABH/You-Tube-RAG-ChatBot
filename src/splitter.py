from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    print("✂️ Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # 🧹 Remove empty / useless chunks
    cleaned_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    print(f"✅ Total chunks after cleaning: {len(cleaned_chunks)}")

    # 🔍 Debug: Show sample chunks
    print("\n🔍 Sample Chunks:\n")
    for i, chunk in enumerate(cleaned_chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:300])
        print()

    print("=" * 60)

    return cleaned_chunks