import streamlit as st
from src.rag_pipeline import RAGPipeline


# -------------------------------
# CLEAN URL FUNCTION
# -------------------------------
def clean_youtube_url(url):
    if "&" in url:
        url = url.split("&")[0]
    return url


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")

st.title("🎥 YouTube RAG Chatbot")
st.markdown("Ask questions from any YouTube video")

# -------------------------------
# SESSION STATE
# -------------------------------
if "rag" not in st.session_state:
    st.session_state.rag = None

if "video_title" not in st.session_state:
    st.session_state.video_title = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------
# INPUT: YOUTUBE URL
# -------------------------------
video_url = st.text_input("🔗 Enter YouTube Video URL")


# -------------------------------
# LOAD VIDEO
# -------------------------------
if st.button("Load Video"):
    if video_url:
        try:
            with st.spinner("🔄 Processing video..."):

                clean_url = clean_youtube_url(video_url)

                # 🔥 Build RAG pipeline (this is stable)
                rag = RAGPipeline(clean_url)

                st.session_state.rag = rag

                # 🔥 SAFE TITLE (no pytube dependency)
                video_id = clean_url.split("v=")[-1]
                st.session_state.video_title = f"YouTube Video ({video_id})"

                # Reset chat
                st.session_state.chat_history = []

            st.success("✅ Video processed successfully!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# -------------------------------
# SHOW VIDEO TITLE
# -------------------------------
if st.session_state.video_title:
    st.markdown("### 🎬 Current Video:")
    st.info(st.session_state.video_title)


# -------------------------------
# CHAT INTERFACE
# -------------------------------
if st.session_state.rag:

    query = st.text_input("💬 Ask a question")

    if st.button("Ask") and query:
        with st.spinner("🤖 Thinking..."):
            answer = st.session_state.rag.query(query)

            # Save chat
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", answer))


# -------------------------------
# DISPLAY CHAT HISTORY
# -------------------------------
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")