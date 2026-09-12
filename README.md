# RAG Question Answering System 🤖

A production-ready Retrieval Augmented Generation (RAG) system built with Transformers, Chroma, and Streamlit.

## Features

✨ **Retrieval**: Vector database search using Chroma + sentence-transformers
✨ **Generation**: LLM-powered answers (OpenAI, Groq, HuggingFace)
✨ **Grounding**: Answers based on your documents with source attribution
✨ **Web UI**: Beautiful Streamlit interface
✨ **Production Ready**: Deployable to Streamlit Cloud

## Project Structure

```
RAG-project/
├── app/
│   ├── rag_app.py       # Main Streamlit application
│   ├── ingest.py        # Builds the vector database from data/documents/
│   └── retrieval.py     # Retrieval helper functions
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── data/
│   ├── documents/       # Your documents, any supported format - edit these
│   └── chroma_db/       # Generated vector database (git-ignored)
├── notebooks/
│   ├── Week1_Embeddings.ipynb
│   ├── Week2_Retrieval.ipynb
│   └── Week3_Generation.ipynb
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/sanjil18/RAG-project.git
cd RAG-project
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the Vector Database

The Chroma database is generated locally and is not committed to git, so you
must build it once before the app can answer anything:

```bash
python -m app.ingest
```

This reads every supported document in `data/documents/` and writes the
embeddings to `data/chroma_db/`. The first run downloads the
`all-MiniLM-L6-v2` embedding model (about 90 MB).

To use your own material, drop the files into `data/documents/` and run the
command again. It rebuilds the collection from scratch each time, and a running app picks
up the new index on its next interaction. Scanned PDFs are pictures with no
text layer, so they need OCR first. Any other file type is skipped, and the
command lists what it skipped.

#### Supported formats

| Format | Extensions | What gets indexed |
|---|---|---|
| PDF | `.pdf` | The text layer. Scanned PDFs need OCR first |
| Word | `.docx` | Paragraphs and tables |
| Excel | `.xlsx`, `.xlsm`, `.xls` | Every sheet, each row stored with its column names |
| PowerPoint | `.pptx` | Slide text, tables, grouped shapes and speaker notes |
| CSV / TSV | `.csv`, `.tsv` | Comma, semicolon, tab or pipe separated |
| Text | `.txt`, `.md`, `.markdown`, `.rst`, `.log` | Everything |
| Web page | `.html`, `.htm` | Visible text, not scripts or styles |
| JSON | `.json`, `.jsonl` | Every value, labelled with its key path |

Old `.doc` and `.ppt` files, RTF, OpenDocument, Apple iWork files, images and
zip archives are listed as skipped, each with a note on how to convert it.
Subfolders are included. Office lock files (`~$...`) are ignored.

### 4. Add a Groq API Key

The app defaults to Groq, which answers in about a second, needs no download
and has a free tier. Get a key at
[console.groq.com/keys](https://console.groq.com/keys), then copy
`.env.example` to `.env` and fill it in:

```
GROQ_API_KEY=gsk_your_key_here
```

The app loads that automatically, so you never paste it into the sidebar. You
can also override the model there with `GROQ_MODEL=...` if Groq retires the
default.

The local HuggingFace option is still available in the sidebar, but
`google/flan-t5-large` needs roughly 3 GB of RAM to load and more while
loading. On a machine with 8 GB it will usually fail with a Windows paging
error. Prefer `google/flan-t5-base` if you want a local model.

### 5. Run Streamlit App

```bash
streamlit run app/rag_app.py
```

Opens at: http://localhost:8501

## Troubleshooting

**The app starts, then the server dies and the browser shows
`ERR_CONNECTION_REFUSED`.** On Windows this is a native crash, not a Python
error, so there is no traceback anywhere.

pandas 2.0.3 bundles its own stale copy of the Visual C++ runtime beside one of
its C extensions:

```
pandas/_libs/window/msvcp140.dll        14.29.30139.0
pandas/_libs/window/vcruntime140_1.dll  14.29.30139.0
```

Windows resolves DLLs by base name, so the first `MSVCP140.dll` loaded into a
process wins for everything after it. Streamlit imports pandas at startup, so
that stale copy loads early. torch, sentencepiece, onnxruntime and scikit-learn
are built against a newer toolchain, bind to it, and the process dies with
access violation `0xc0000005`. Windows Event Viewer records it under
**Application Error** with `MSVCP140.dll` as the faulting module.

Fix it by renaming those bundled DLLs so Windows falls back to the system
runtime:

```bash
python -m app._winruntime
```

Re-run that after any `pip install`, which can restore them. Use
`python -m app._winruntime --restore` to undo.

**"The knowledge base is not ready yet".** The database has no indexed
documents. Run `python -m app.ingest`.

**I added a file but the app cannot answer questions about it.** Adding a
file does not index it. Run `python -m app.ingest` after every change to
`data/documents/`. Check its output lists your file. If it says the file was
skipped, it is an unsupported type or a scanned PDF with no text.

**"The paging file is too small for this operation to complete" (error 1455).**
You picked the local HuggingFace provider and `google/flan-t5-large` does not
fit in available memory. It needs about 3 GB for weights and roughly the same
again while loading. Use Groq, or switch to `google/flan-t5-base`.

**Groq fails with a model decommissioned error.** Groq retires models
periodically. Put a current one from
[console.groq.com/docs/models](https://console.groq.com/docs/models) into the
"Groq model" box in the sidebar, or set `GROQ_MODEL` in `.env`.

**`TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`.**
The `groq` client is too old for the installed `httpx`. Run
`pip install -U groq`.

## How It Works

```
User Question
    ↓
[RETRIEVAL] Search Chroma DB for relevant chunks
    ↓
[PROMPT] Format: "Using these docs, answer: {question}"
    ↓
[GENERATION] LLM generates grounded answer
    ↓
Answer + Source Documents
```

## LLM Options

### HuggingFace (Default - Free)
- No API key needed
- Free, runs locally
- Speed: 5-10 seconds/query

### OpenAI (Best Quality)
- API key required
- Speed: 2-3 seconds/query
- Cost: ~$0.0005/query

### Groq (Free + Fast)
- API key required (free tier)
- Speed: ~500ms/query
- Completely free

## Technologies

- **ML/NLP**: Transformers, Sentence-Transformers
- **Vector DB**: Chroma
- **Web**: Streamlit
- **LLMs**: OpenAI, Groq, HuggingFace

## Deployment

### Streamlit Cloud (5 minutes)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect your repo
4. Select: `app/rag_app.py`
5. Deploy!

## Future Improvements

- [ ] Document upload UI
- [ ] Re-ranking
- [ ] Query expansion
- [ ] Feedback loop
- [ ] API endpoint
- [ ] Docker deployment

## License

MIT License - see LICENSE file

## Author

Sanji Raj - [GitHub](https://github.com/sanjil18)
