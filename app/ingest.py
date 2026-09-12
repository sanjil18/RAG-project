"""
Build the Chroma index from the documents in data/documents/.

Run this once before starting the app (and again whenever you add, edit or
remove a document):

    python -m app.ingest

It reads every supported document in the documents folder, subfolders
included, and writes the
embeddings into data/chroma_db/ as the "rag_collection" collection, which is
what app/rag_app.py reads at query time.
"""

import argparse
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must run before any native library loads. See app/_winruntime.py.
from app._winruntime import preload_system_msvc_runtime

preload_system_msvc_runtime()

from app.retrieval import CHROMA_FOLDER, COLLECTION_NAME, DOCUMENTS_FOLDER
from app.loaders import FRIENDLY_FORMATS, UnsupportedFormatError, read_document

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ----------------------------------------------------------------------
# Text splitting
# ----------------------------------------------------------------------
# A self-contained port of LangChain's RecursiveCharacterTextSplitter, using
# the same settings the original notebook used. Implemented here so the app
# does not need langchain-text-splitters as a dependency.


def _join_splits(splits: List[str], separator: str) -> str:
    text = separator.join(splits)
    return text.strip()


def _merge_splits(
    splits: List[str], separator: str, chunk_size: int, chunk_overlap: int
) -> List[str]:
    """Greedily pack splits into chunks of <= chunk_size, keeping an overlap."""
    sep_len = len(separator)
    chunks: List[str] = []
    current: List[str] = []
    total = 0

    for split in splits:
        split_len = len(split)
        # Would adding this split overflow the current chunk?
        if current and total + split_len + (sep_len if current else 0) > chunk_size:
            merged = _join_splits(current, separator)
            if merged:
                chunks.append(merged)
            # Drop from the front until the tail fits under the overlap budget.
            while current and (
                total > chunk_overlap
                or total + split_len + (sep_len if current else 0) > chunk_size
            ):
                total -= len(current[0]) + (sep_len if len(current) > 1 else 0)
                current.pop(0)
        current.append(split)
        total += split_len + (sep_len if len(current) > 1 else 0)

    merged = _join_splits(current, separator)
    if merged:
        chunks.append(merged)
    return chunks


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = None,
) -> List[str]:
    """Recursively split text on the first separator that keeps chunks small."""
    separators = list(separators if separators is not None else SEPARATORS)

    # Pick the finest separator that actually occurs in this text.
    separator = separators[-1]
    remaining: List[str] = []
    for i, sep in enumerate(separators):
        if sep == "":
            separator = sep
            remaining = []
            break
        if sep in text:
            separator = sep
            remaining = separators[i + 1 :]
            break

    splits = list(text) if separator == "" else text.split(separator)
    splits = [s for s in splits if s != ""]

    good: List[str] = []
    final: List[str] = []
    for split in splits:
        if len(split) < chunk_size:
            good.append(split)
            continue
        # Too big on its own: flush what we have, then recurse into it.
        if good:
            final.extend(_merge_splits(good, separator, chunk_size, chunk_overlap))
            good = []
        if remaining:
            final.extend(split_text(split, chunk_size, chunk_overlap, remaining))
        else:
            final.append(split)
    if good:
        final.extend(_merge_splits(good, separator, chunk_size, chunk_overlap))
    return [c for c in final if c.strip()]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


# Folders inside the documents folder that are never documents.
SKIP_DIRS = {"chroma_db", "__pycache__", ".git", ".ipynb_checkpoints"}


def load_documents(folder: str) -> Dict[str, str]:
    """Read every supported document under folder, including subfolders.

    Keys are paths relative to folder with the extension kept, so report.pdf
    and report.docx stay separate documents and the app shows exactly which
    file an answer came from.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Documents folder does not exist: {folder}")

    docs: Dict[str, str] = {}
    skipped: List[str] = []
    failed: List[str] = []

    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for filename in sorted(files):
            # "~$Report.docx" is the lock file Office keeps while a document is
            # open. It is not a document and cannot be parsed.
            if filename.startswith(("~$", ".")):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            try:
                text = read_document(path).strip()
            except UnsupportedFormatError as exc:
                skipped.append(f"{rel}: {exc}")
                continue
            except Exception as exc:  # one bad file must not stop the rest
                failed.append(f"{rel}: {type(exc).__name__}: {exc}")
                continue
            if not text:
                reason = (
                    "no text layer, probably a scanned PDF that needs OCR"
                    if rel.lower().endswith(".pdf")
                    else "no text found"
                )
                skipped.append(f"{rel}: {reason}")
                continue
            docs[rel] = text

    # Say what was left out. A silently skipped file is how a new document
    # ends up "not answering" with no clue why.
    if skipped:
        print(f"  Skipped {len(skipped)} file(s):")
        for line in skipped:
            print(f"    - {line}")
    if failed:
        print(f"  Could not read {len(failed)} file(s):")
        for line in failed:
            print(f"    - {line}")
    return docs


# ----------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------


def build_index(
    documents_folder: str = DOCUMENTS_FOLDER,
    chroma_folder: str = CHROMA_FOLDER,
    reset: bool = True,
) -> int:
    """Chunk every document and (re)build the Chroma collection. Returns count."""
    documents = load_documents(documents_folder)
    if not documents:
        raise ValueError(
            f"No readable documents found in {documents_folder}. "
            f"Supported formats: {FRIENDLY_FORMATS}."
        )

    print(f"Found {len(documents)} document(s) in {documents_folder}")

    all_chunks: List[str] = []
    chunk_metadata: List[Dict] = []

    for doc_name, doc_text in documents.items():
        chunks = split_text(doc_text)
        if not chunks:
            print(f"  {doc_name}: no usable text, skipped")
            continue
        print(
            f"  {doc_name}: {len(doc_text)} chars -> {len(chunks)} chunks "
            f"(avg {len(doc_text) // len(chunks)} chars)"
        )
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append(
                {
                    "source": doc_name,
                    "chunk_id": i,
                    "chunk_number": f"{i + 1}/{len(chunks)}",
                }
            )

    if not all_chunks:
        raise ValueError("Documents produced no chunks. Nothing to index.")

    os.makedirs(chroma_folder, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_folder)

    # Use the same embedding function the app uses at query time, so the
    # vectors written here match the vectors queries are compared against.
    # This also keeps us off chromadb's default ONNX function, whose runtime
    # fails to load on this machine.
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads ~90 MB)")
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Embedding and adding {len(all_chunks)} chunks...")
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        documents=all_chunks,
        metadatas=chunk_metadata,
    )

    count = collection.count()
    print(f"Done. Collection '{COLLECTION_NAME}' now holds {count} chunks.")
    print(f"Stored in: {os.path.abspath(chroma_folder)}")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents", default=DOCUMENTS_FOLDER, help="Folder of source documents"
    )
    parser.add_argument(
        "--chroma", default=CHROMA_FOLDER, help="Folder for the Chroma database"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing collection instead of rebuilding it",
    )
    args = parser.parse_args()

    try:
        build_index(args.documents, args.chroma, reset=not args.append)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
