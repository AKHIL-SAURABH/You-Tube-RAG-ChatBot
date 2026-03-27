from src.rag_pipeline import RAGPipeline

video_url = input("Enter YouTube URL: ")

rag = RAGPipeline(video_url)

while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        print("Exiting...")
        break
    
    answer = rag.query(query)
    print("\nAnswer:", answer)