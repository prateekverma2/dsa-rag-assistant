import os
import sys
import json
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone

# 🧩 Ensure project root in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Load environment variables
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Import pipeline
from src.core.rag_pipeline import RAGPipeline



def get_pinecone_stats():
    """Fetch current Pinecone vector stats (total chunks in namespace)."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX"))
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        namespace = os.getenv("PINECONE_NAMESPACE", "main")
        count = namespaces.get(namespace, {}).get("vector_count", 0)
        total = stats.get("total_vector_count", 0)
        return count, total
    except Exception as e:
        print(f"⚠️ Pinecone stats error: {e}")
        return 0, 0



st.set_page_config(page_title="🧠 DSA Tutor RAG", layout="wide")
st.title("🧠 DSA Tutor — Retrieval-Augmented DSA Assistant")


if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(mode="cloud")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []     # [{query, answer}, ...]
if "messages" not in st.session_state:
    st.session_state.messages = []         # [{role, content}, ...]
if "learn_log" not in st.session_state:
    st.session_state.learn_log = []        # recent auto-learn events
if "chunk_count" not in st.session_state or "total_count" not in st.session_state:
    st.session_state.chunk_count, st.session_state.total_count = get_pinecone_stats()

pipeline = st.session_state.pipeline


st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Model Mode:",
    ["cloud (Groq)", "local (Ollama)"],
    index=0 if pipeline.mode == "cloud" else 1,
    help="Choose Groq Cloud API or Local Ollama model",
)
selected_mode = "cloud" if "groq" in mode.lower() else "local"
if selected_mode != pipeline.mode:
    pipeline.mode = selected_mode
    st.sidebar.success(f"✅ Switched to {selected_mode.upper()}")

# Top-K affects retrieval immediately
top_k = st.sidebar.slider("Top K context chunks", 1, 10, pipeline.top_k if hasattr(pipeline, "top_k") else 5)
if hasattr(pipeline, "top_k"):
    pipeline.top_k = int(top_k)

st.sidebar.divider()


st.sidebar.subheader("💾 Chat History Manager")


CHAT_DIR = os.path.join(os.getcwd(), "chat_history")
os.makedirs(CHAT_DIR, exist_ok=True)
pipeline.chat_dir = CHAT_DIR  # sync with pipeline


saved_chats = sorted([f for f in os.listdir(CHAT_DIR) if f.endswith(".json")], reverse=True)

# Dropdown to select old chat
selected_chat = st.sidebar.selectbox("📂 Select a saved chat", options=["(none)"] + saved_chats)

# --- Button Row ---
col1, col2, col3 = st.sidebar.columns([1, 1, 1])

#  SAVE CHAT
with col1:
    if st.button("💾 Save"):
        if not st.session_state.chat_history:
            st.sidebar.warning("⚠️ No chat to save yet.")
        else:
            fname = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = os.path.join(CHAT_DIR, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.chat_history, f, indent=4)
            st.sidebar.success(f"✅ Chat saved as {fname}")
            st.rerun()

#  LOAD CHAT
with col2:
    if st.button("📂 Load") and selected_chat != "(none)":
        try:
            path = os.path.join(CHAT_DIR, selected_chat)
            with open(path, "r", encoding="utf-8") as f:
                loaded_chat = json.load(f)

            if isinstance(loaded_chat, list) and loaded_chat:
                # restore conversation + memory
                st.session_state.chat_history = loaded_chat
                pipeline.chat_history = loaded_chat.copy()
                st.session_state.messages = []
                for turn in loaded_chat:
                    st.session_state.messages.append({"role": "user", "content": turn.get("query", "")})
                    st.session_state.messages.append({"role": "assistant", "content": turn.get("answer", "")})

                st.sidebar.success(f"✅ Loaded chat: {selected_chat}")
                st.rerun()
            else:
                st.sidebar.warning("⚠️ Selected chat file is empty or invalid.")
        except Exception as e:
            st.sidebar.error(f"❌ Load failed: {e}")

#  DELETE CHAT
with col3:
    if st.button("🗑️ Delete") and selected_chat != "(none)":
        os.remove(os.path.join(CHAT_DIR, selected_chat))
        st.sidebar.warning(f"🗑️ Deleted {selected_chat}")
        st.rerun()

#  RESET CHAT
if st.sidebar.button("🧹 New Chat (Reset)"):
    st.session_state.chat_history = []
    st.session_state.messages = []
    pipeline.chat_history.clear()
    st.sidebar.info("🧼 Conversation reset successfully!")

st.sidebar.divider()

st.sidebar.markdown("📘 **Auto-Learn Status**")
st.sidebar.info(
    "✅ Approved → Promoted to Pinecone index\n\n"
    "⚠️ Unsure → Queued for manual review\n\n"
    "❌ Rejected → Added to learn queue"
)

st.sidebar.divider()
st.sidebar.markdown("📊 **Pinecone Index Status**")

# Show live stats (also keep session copies updated on refresh)
live_chunk_count, live_total_count = get_pinecone_stats()
st.sidebar.metric("🧮 Chunks in Namespace", f"{live_chunk_count:,}")
st.sidebar.metric("🌍 Total Vectors in Index", f"{live_total_count:,}")

if st.sidebar.button("🔄 Refresh Pinecone Stats"):
    st.session_state.chunk_count, st.session_state.total_count = get_pinecone_stats()
    st.sidebar.success("✅ Pinecone stats refreshed.")

# 🧩 Validator Debug Mode
st.sidebar.divider()
debug_mode = st.sidebar.toggle("🧠 Enable Validator Debug Mode", value=False)
st.session_state["debug_mode"] = debug_mode


st.subheader("💬 Chat with your DSA Tutor")

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Input & response
if prompt := st.chat_input("Ask your question here..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            out = pipeline.ask(prompt)
            response = out["answer"]
            st.markdown(response)

            # Optional: show validator raw output if debug enabled
            if st.session_state.get("debug_mode")   and "validator_raw" in out:
                st.session_state.debug_data = out["validator_raw"]
            else:
                st.session_state.debug_data = None


            # Keep structured history (for saving & memory)
            st.session_state.chat_history.append({"query": prompt, "answer": response})
            pipeline.chat_history = st.session_state.chat_history.copy()

            # 🧠 Log Auto-Learn only when validator actually RAN & APPROVED
            if out.get("validated") and out.get("verdict") == "approve":
                st.session_state.learn_log.append({
                    "question": prompt,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "status": "Promoted to Pinecone",
                    "chunks_added": len(out.get("context_chunks", []))
                })
            else:
                # helpful when you're debugging skip/ran behavior
                print("⚡ Validation skipped — no auto-learn log recorded.")

    st.session_state.messages.append({"role": "assistant", "content": response})
    # Update stats after potential auto-learn
    st.session_state.chunk_count, st.session_state.total_count = get_pinecone_stats()
    st.rerun()


if st.session_state.learn_log:
    st.sidebar.divider()
    st.sidebar.markdown("🧠 **Recent Auto-Learn Events**")
    for item in reversed(st.session_state.learn_log[-5:]):
        st.sidebar.markdown(
            f"**{item['timestamp']}** → {item['status']}<br>"
            f"📝 *{item['question'][:50]}...*<br>"
            f"📦 {item['chunks_added']} chunks added",
            unsafe_allow_html=True
        )




st.markdown("---")
st.caption("— DSA Tutor RAG System (Groq + Ollama Powered) 🚀")
