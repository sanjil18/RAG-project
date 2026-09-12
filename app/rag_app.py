"""
RAG Question Answering System - Streamlit Web App
Complete web interface for RAG built in Weeks 1-3
"""

import os
import sys
from datetime import datetime

# Add parent directory to path to import retrieval
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This has to happen before `import streamlit`, because streamlit imports
# pandas, and pandas loads a stale bundled msvcp140.dll that makes every later
# native import (torch, sentencepiece, onnxruntime) crash the process with a
# Windows access violation. See app/_winruntime.py. Do not move it below the
# streamlit import.
from app._winruntime import preload_system_msvc_runtime

preload_system_msvc_runtime()

# Load API keys from .env so they do not have to be pasted every session.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
except Exception:
    pass

import streamlit as st

from app.retrieval import (
    CollectionMissingError,
    DATA_FOLDER,
    DOCUMENTS_FOLDER,
    build_prompt,
    get_chroma_collection,
    retrieve_documents,
)
from app.loaders import FRIENDLY_FORMATS

# ========================================================================
# PAGE CONFIGURATION
# ========================================================================

st.set_page_config(
    page_title="RAG QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CUSTOM CSS FOR BETTER STYLING
# ========================================================================

st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    .section-header {
        color: #1f77b4;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }

    .answer-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 20px 0;
    }

    .source-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #28a745;
    }

    .score-badge {
        display: inline-block;
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9rem;
    }

    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 15px 0;
    }

    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 15px 0;
    }

    .stat-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================================
# SIDEBAR - CONFIGURATION & SETTINGS
# ========================================================================

# Default models. Defined before the sidebar, which lets you override them.
# Groq retires models often: mixtral-8x7b-32768 and llama-3.3-70b-versatile
# both return 404 now. Checked against the live model list on 2026-09-11.
GROQ_MODEL = "openai/gpt-oss-120b"
HUGGINGFACE_MODEL = "google/flan-t5-large"

st.sidebar.markdown("## ⚙️ Configuration")

_FOLDER_KEY = "rag_folder_path"

# A stable key keeps this box's value in session state, which also means a bad
# value sticks around across reruns. The reset button below is the way out.
if _FOLDER_KEY not in st.session_state:
    st.session_state[_FOLDER_KEY] = DATA_FOLDER

rag_folder = st.sidebar.text_input(
    "RAG Folder Path",
    key=_FOLDER_KEY,
    help="Folder that contains chroma_db. Should normally be the project's data folder.",
)


def _reset_folder():
    st.session_state[_FOLDER_KEY] = DATA_FOLDER


if os.path.abspath(rag_folder or "") != os.path.abspath(DATA_FOLDER):
    st.sidebar.button(
        "↩️ Reset to default folder", on_click=_reset_folder, use_container_width=True
    )

rag_folder = os.path.expanduser((rag_folder or "").strip().strip('"'))

# Point the box at a file rather than a folder and every later path is nonsense,
# so say so here instead of letting Chroma report a confusing WinError 3.
folder_problem = None
if not rag_folder:
    folder_problem = "The RAG folder path is empty."
elif os.path.isfile(rag_folder):
    folder_problem = (
        f"The RAG folder path points at a file, not a folder:\n\n`{rag_folder}`\n\n"
        f"It should be the folder that contains `chroma_db`."
    )
elif not os.path.isdir(rag_folder):
    folder_problem = f"The RAG folder path does not exist:\n\n`{rag_folder}`"

chroma_folder = os.path.join(rag_folder, "chroma_db") if rag_folder else ""

st.sidebar.markdown("---")

# Retrieval settings
st.sidebar.markdown("### 🔍 Retrieval Settings")

top_k = st.sidebar.slider(
    "Number of chunks to retrieve",
    min_value=1,
    max_value=10,
    value=3,
    help="How many relevant documents to retrieve"
)

similarity_threshold = st.sidebar.slider(
    "Relevance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum relevance score to include a document"
)

st.sidebar.markdown("---")

# LLM settings
st.sidebar.markdown("### 🤖 LLM Settings")

llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    options=["Groq (Free, Fast)", "OpenAI (Requires API key)", "HuggingFace (Free, Local)"],
    index=0,
    help="Select which LLM to use for generation"
)

def _hf_model_is_cached(repo_id: str) -> bool:
    """True if the model's weights are already on disk."""
    try:
        from huggingface_hub import constants as _hf_constants

        # Renamed across huggingface_hub versions: HUGGINGFACE_HUB_CACHE in
        # 0.17.x, HF_HUB_CACHE in newer releases.
        cache_root = getattr(_hf_constants, "HF_HUB_CACHE", None) or getattr(
            _hf_constants, "HUGGINGFACE_HUB_CACHE"
        )
        folder = os.path.join(cache_root, "models--" + repo_id.replace("/", "--"))
        snapshots = os.path.join(folder, "snapshots")
        if not os.path.isdir(snapshots):
            return False
        for root, _dirs, files in os.walk(snapshots):
            if any(f.endswith((".safetensors", ".bin")) for f in files):
                return True
        return False
    except Exception:
        return False


if "HuggingFace" in llm_provider:
    if _hf_model_is_cached("google/flan-t5-large"):
        st.sidebar.caption(
            "✅ google/flan-t5-large is downloaded. It runs locally on your CPU, "
            "so answers take a while but cost nothing. The API options are much "
            "faster."
        )
    else:
        st.sidebar.caption(
            "⏬ The first answer downloads google/flan-t5-large (about 3 GB) and "
            "runs it on your CPU. Expect a long wait the first time, and do not "
            "touch the controls while it downloads. The API options answer in "
            "seconds and download nothing."
        )

temperature = st.sidebar.slider(
    "Temperature (Creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Higher = more creative, Lower = more focused"
)

st.sidebar.markdown("---")

# API Keys (if needed)
api_key = None
groq_model = GROQ_MODEL
if "OpenAI" in llm_provider:
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get from https://platform.openai.com/api-keys",
    )
elif "Groq" in llm_provider:
    api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get from https://console.groq.com/keys",
    )
    if os.getenv("GROQ_API_KEY"):
        st.sidebar.caption("🔑 Key loaded from your .env file.")
    else:
        st.sidebar.caption(
            "Get a free key at console.groq.com/keys. Put it in a `.env` file as "
            "`GROQ_API_KEY=...` to skip pasting it every time."
        )

    # Groq retires models periodically, so keep this editable rather than
    # hard-coded. If an answer comes back as a model error, pick another from
    # console.groq.com/docs/models.
    groq_model = st.sidebar.text_input(
        "Groq model",
        value=os.getenv("GROQ_MODEL", GROQ_MODEL),
        help="Change this if Groq reports the model is decommissioned.",
    )

st.sidebar.markdown("---")

# Information
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("""
**RAG Q&A System**

This system uses:
- 📚 Retrieval: Search relevant documents
- 🤖 Generation: Answer using LLM
- 🎯 Grounding: Answers based on documents

**Settings:**
- Top-k: Number of docs to retrieve
- Threshold: Minimum relevance score
- Temperature: Answer creativity
""")

# ========================================================================
# LOAD COMPONENTS (CACHING FOR PERFORMANCE)
# ========================================================================

@st.cache_resource
def load_chroma_db(folder: str, db_stamp: float = 0.0):
    """Load Chroma database (cached for performance).

    db_stamp is the database file's modification time. It is part of the
    cache key, so re-running `python -m app.ingest` while the app is open is
    picked up on the next interaction, instead of the app querying a
    collection that ingest has since deleted. Opening and querying the
    database do not change that timestamp, so this never rebuilds per question.

    Returns (collection, error_message). Exactly one of the two is set, so the
    caller can tell "not built yet" apart from "something else went wrong".
    """
    try:
        return get_chroma_collection(folder), None
    except CollectionMissingError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Could not open the Chroma database in {folder}: {e}"

@st.cache_resource
def load_embedding_model():
    """Load embedding model (cached for performance)"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None



# Cached on (provider, api_key), so switching provider still rebuilds, but
# asking a second question reuses what is already in memory.
#
# Without this cache every rerun built a brand new pipeline. Streamlit reruns
# on any widget interaction, so a single stray click during the first load
# started a SECOND 3 GB download of the same model alongside the first. The
# key checks stay outside, so a "no API key yet" result is never cached.
@st.cache_resource(show_spinner="Loading language model...")
def _build_llm(provider: str, api_key: str):
    """Build the LLM client or pipeline for a provider. Expensive; cached."""
    if "HuggingFace" in provider:
        from transformers import pipeline

        # Downloads about 3 GB the first time, then loads from local cache.
        return pipeline("text2text-generation", model=HUGGINGFACE_MODEL)
    if "OpenAI" in provider:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    if "Groq" in provider:
        from groq import Groq

        return Groq(api_key=api_key)
    return None


def load_llm():
    """Load LLM based on selected provider"""
    if "HuggingFace" not in llm_provider and not api_key:
        st.warning(f"Please enter your {llm_provider.split('(')[0].strip()} API key in the sidebar")
        return None
    try:
        return _build_llm(llm_provider, api_key or "")
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

# ========================================================================
# GENERATE FUNCTION
# ========================================================================

def generate_answer(prompt: str, llm, provider: str) -> str:
    """Generate answer using selected LLM"""
    try:
        if "HuggingFace" in provider:
            result = llm(prompt, max_length=500, do_sample=True, temperature=max(temperature, 0.01))
            return result[0]['generated_text']

        elif "OpenAI" in provider:
            response = llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=temperature
            )
            return response.choices[0].message.content

        elif "Groq" in provider:
            message = llm.chat.completions.create(
                model=groq_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return message.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"

# ========================================================================
# MAIN APP LAYOUT
# ========================================================================

# Header
st.markdown('<p class="main-title">🤖 RAG Question Answering System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ask questions about your documents. Get grounded answers with sources.</p>', unsafe_allow_html=True)

st.markdown("---")

# ========================================================================
# CHECK IF COMPONENTS ARE LOADED
# ========================================================================

# A bad folder in the sidebar is a different problem from an unbuilt database,
# and it has a different fix, so handle it first.
if folder_problem:
    st.error("❌ The RAG Folder Path in the sidebar is not usable.")
    st.warning(folder_problem)
    st.markdown(f"**It should be:** `{DATA_FOLDER}`")
    st.button(
        "↩️ Reset it for me", on_click=_reset_folder, type="primary", key="reset_main"
    )
    st.stop()

# Load components
_db_file = os.path.join(chroma_folder, "chroma.sqlite3")
_db_stamp = os.path.getmtime(_db_file) if os.path.exists(_db_file) else 0.0
collection, db_error = load_chroma_db(chroma_folder, _db_stamp)

if collection is None:
    st.error("❌ The knowledge base is not ready yet.")
    st.warning(db_error)
    st.markdown("**To build it, run this in your terminal from the project root:**")
    st.code("python -m app.ingest", language="bash")
    st.caption(
        f"That indexes every supported document ({FRIENDLY_FORMATS}) in {DOCUMENTS_FOLDER} "
        f"and writes the vectors to {chroma_folder}. Reload this page when it finishes."
    )
    st.stop()

embedding_model = load_embedding_model()
if embedding_model is None:
    st.error("❌ Error: Could not load embedding model.")
    st.stop()

collection_size = collection.count()
st.success(f"✅ Connected to knowledge base with {collection_size} documents")

# ========================================================================
# MAIN INTERFACE
# ========================================================================

# Question input
st.markdown('<p class="section-header">📝 Your Question</p>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])

with col1:
    user_query = st.text_input(
        "Enter your question:",
        placeholder="What is a transformer? How does attention work? etc.",
        label_visibility="collapsed"
    )

with col2:
    submit_button = st.button("🔍 Search", use_container_width=True)

# ========================================================================
# PROCESS QUERY
# ========================================================================

if submit_button and user_query:
    with st.spinner("🔄 Processing your question..."):

        # Step 1: Retrieve
        st.info("Step 1/3: Retrieving relevant documents...")
        retrieved = retrieve_documents(user_query, collection, top_k, similarity_threshold)

        if retrieved is None:
            st.error("Error during retrieval")
            st.stop()

        retrieved_docs = retrieved['documents']
        if not retrieved_docs:
            st.warning("⚠️ No relevant documents found. Answer may not be grounded.")

        # Step 2: Build prompt
        st.info("Step 2/3: Building prompt...")
        prompt = build_prompt(user_query, retrieved_docs)

        # Step 3: Generate
        st.info("Step 3/3: Generating answer...")
        llm = load_llm()

        if llm is None and "HuggingFace" not in llm_provider:
            st.error("Error loading LLM. Please check API keys in sidebar.")
            st.stop()

        answer = generate_answer(prompt, llm, llm_provider)

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    st.markdown('<p class="section-header">💡 Answer</p>', unsafe_allow_html=True)

    # st.info renders the model's markdown (bold, lists) properly. Injecting the
    # answer into a raw HTML div showed literal ** asterisks, and let model
    # output inject arbitrary HTML into the page.
    st.info(answer, icon="💡")

    # Display sources
    st.markdown('<p class="section-header">📚 Source Documents</p>', unsafe_allow_html=True)

    if retrieved['documents']:
        for i, (chunk, score, source) in enumerate(zip(
            retrieved['documents'],
            retrieved['scores'],
            retrieved['sources']
        ), 1):
            with st.expander(
                f"📄 Document {i}: {source} - Score: {score:.2%}",
                expanded=(i == 1)
            ):
                st.write(chunk)
                st.caption(f"Relevance Score: {score:.3f} | Source: {source}")
    else:
        st.warning("No documents retrieved. Results may not be grounded.")

    # Display statistics
    st.markdown('<p class="section-header">📊 Statistics</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Documents Retrieved", len(retrieved['documents']))

    with col2:
        avg_score = sum(retrieved['scores']) / len(retrieved['scores']) if retrieved['scores'] else 0
        st.metric("Avg Relevance", f"{avg_score:.2%}")

    with col3:
        st.metric("LLM Provider", llm_provider.split("(")[0].strip())

    with col4:
        st.metric("Temperature", temperature)

    # Quality assessment
    st.markdown('<p class="section-header">🎯 Quality Assessment</p>', unsafe_allow_html=True)

    if len(retrieved['documents']) == 0:
        st.error("❌ No relevant documents found. Answer reliability is LOW.")
    elif min(retrieved['scores']) < 0.3:
        st.warning("⚠️ Low relevance scores. Answer may not be well-grounded.")
    elif sum(retrieved['scores']) / len(retrieved['scores']) >= 0.7:
        st.success("✅ High quality retrieval. Answer is well-grounded in documents.")
    else:
        st.info("ℹ️ Moderate quality retrieval. Answer has reasonable grounding.")

    # Save to history
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        'timestamp': datetime.now().isoformat(),
        'query': user_query,
        'answer': answer,
        'sources': retrieved['sources'],
        'scores': retrieved['scores']
    })

# ========================================================================
# CONVERSATION HISTORY
# ========================================================================

if 'history' in st.session_state and st.session_state.history:
    st.markdown("---")
    st.markdown('<p class="section-header">📋 Conversation History</p>', unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

    for i, item in enumerate(reversed(st.session_state.history), 1):
        with st.expander(f"Query {len(st.session_state.history) - i + 1}: {item['query'][:50]}..."):
            st.write(f"**Question:** {item['query']}")
            st.write(f"**Answer:** {item['answer'][:200]}...")
            st.write(f"**Sources:** {', '.join(set(item['sources']))}")
            st.write(f"**Scores:** {[f'{s:.2f}' for s in item['scores']]}")

# ========================================================================
# FOOTER
# ========================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 30px;">
    <p>🤖 RAG Question Answering System</p>
    <p>Built with <strong>Transformers</strong> • <strong>Chroma</strong> • <strong>Streamlit</strong></p>
    <p><a href="https://github.com/sanjil18/RAG-project">GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
