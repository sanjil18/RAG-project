"""Retrieval functions for RAG system"""

import os
import sys

# Load the system MSVC runtime before chromadb pulls in any native library.
# Without this, pandas' stale bundled msvcp140.dll wins the process and the
# next native import dies with a Windows access violation that takes the whole
# Streamlit server down. See app/_winruntime.py for the full explanation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app._winruntime import preload_system_msvc_runtime

preload_system_msvc_runtime()

import importlib
import types

# chromadb builds its default embedding function (ONNXMiniLM_L6_V2) eagerly,
# as a class-level default argument, the moment `chromadb` is imported. That
# constructor imports onnxruntime and raises if it is missing, so an
# unimportable onnxruntime takes `import chromadb` down with it - even though
# this app never uses it, since we always pass our own
# SentenceTransformerEmbeddingFunction.
#
# Stub whichever of these cannot be imported, so `import chromadb` succeeds
# regardless. chromadb only stores the module and would call into it solely if
# the ONNX embedding function were used, which never happens here.
for _name, _attrs in (
    ("onnxruntime", {}),
    ("tokenizers", {"Tokenizer": object}),
    ("tqdm", {"tqdm": lambda *a, **k: None}),
):
    if _name not in sys.modules:
        try:
            importlib.import_module(_name)
        except Exception:
            _stub = types.ModuleType(_name)
            _stub.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
            for _attr, _val in _attrs.items():
                setattr(_stub, _attr, _val)
            sys.modules[_name] = _stub

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import Dict, List

# Shared locations, so the app and the ingestion script always agree.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(_PROJECT_ROOT, "data")
DOCUMENTS_FOLDER = os.path.join(DATA_FOLDER, "documents")
CHROMA_FOLDER = os.path.join(DATA_FOLDER, "chroma_db")
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class CollectionMissingError(RuntimeError):
    """Raised when the Chroma database exists but holds no indexed documents."""


def get_chroma_collection(chroma_folder: str):
    """
    Initialize and return Chroma collection

    Args:
        chroma_folder: Path to chroma_db folder

    Returns:
        Chroma collection object

    Raises:
        CollectionMissingError: the database has not been built yet.
    """
    chroma_client = chromadb.PersistentClient(path=chroma_folder)
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    existing = {c.name for c in chroma_client.list_collections()}
    if COLLECTION_NAME not in existing:
        raise CollectionMissingError(
            f"No '{COLLECTION_NAME}' collection in {os.path.abspath(chroma_folder)}. "
            f"The database is empty - it has never been built on this machine. "
            f"Run `python -m app.ingest` to index the documents in "
            f"{DOCUMENTS_FOLDER}, then reload this page."
        )

    return chroma_client.get_collection(
        COLLECTION_NAME, embedding_function=embedding_function
    )


def retrieve_documents(
    query: str,
    collection,
    top_k: int = 3,
    min_score: float = 0.3
) -> Dict:
    """
    Retrieve relevant documents from vector DB

    Args:
        query: User question
        collection: Chroma collection
        top_k: Number of documents to retrieve
        min_score: Minimum relevance score

    Returns:
        dict: Retrieved documents with scores and sources
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Convert distances to similarities (1 - distance)
        similarities = [1 - d for d in results['distances'][0]]

        filtered = {
            "documents": [],
            "scores": [],
            "sources": [],
            "metadata": []
        }

        for doc, score, meta in zip(
            results['documents'][0],
            similarities,
            results['metadatas'][0]
        ):
            if score >= min_score:
                filtered["documents"].append(doc)
                filtered["scores"].append(score)
                filtered["sources"].append(meta.get('source', 'Unknown'))
                filtered["metadata"].append(meta)

        return filtered

    except Exception as e:
        print(f"Retrieval error: {e}")
        return None


def build_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Build prompt for LLM

    Args:
        query: User question
        context_chunks: List of relevant document chunks

    Returns:
        str: Formatted prompt
    """
    if not context_chunks:
        context_text = "[No relevant context found in documents]"
    else:
        context_text = "\n\n".join(
            [f"Document {i+1}:\n{chunk}"
             for i, chunk in enumerate(context_chunks)]
        )

    prompt = f"""You are a helpful AI assistant. Answer the question using ONLY the provided context.
If the answer is not in the context, clearly say "I don't have enough information to answer this question."

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    return prompt
