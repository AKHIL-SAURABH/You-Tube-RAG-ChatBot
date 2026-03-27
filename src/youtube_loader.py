from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url


def load_youtube_video(url):
    print("📺 Loading YouTube transcript (direct API)...")

    try:
        video_id = extract_video_id(url)

        # ✅ NEW API USAGE
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)

        text = " ".join([t.text for t in transcript])

        print("✅ Transcript loaded successfully!")
        print(f"📄 Transcript length: {len(text)} characters")

        # 🔍 Preview
        print("\n🔍 Transcript Preview:\n")
        print(text[:500])
        print("\n" + "=" * 60)

        from langchain_core.documents import Document
        return [Document(page_content=text)]

    except Exception as e:
        print("❌ Error loading transcript:", str(e))
        return []