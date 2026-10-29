# Document Intelligence RAG

Multi-document question answering system: retrieves evidence from many documents, links facts across sources, detects conflicting information, abstains when evidence is insufficient, and cites its sources.

This is **not** a "chat with PDF" app. A question may require several sources at once (e.g. "which change caused the March payment delays, which services were affected, and which documents confirm it?").

## Features

- **Ingestion:** 4 formats (JSONL, PDF, DOCX, Markdown/TXT), metadata extraction, 2 chunking strategies, 2-level deduplication, create / update / delete (hash-based).
- **Hybrid search:** dense (bge-small embeddings + pgvector HNSW) + sparse (own BM25 implementation), fused with own RRF implementation, plus a title-match re-ranker.
- **RAG:** query decomposition, multi-source retrieval, answer with citations, citation verification, conflict detection, abstention.
- **Tool calling:** agent mode with `search_documents` and `get_document` tools (JSON-based tool protocol, validated by our code).
- **Backend:** FastAPI. **UI:** Streamlit (local) and Gradio (Hugging Face Space).
- **Evaluation:** 50 questions, 3 configurations, Recall, Both-found, Answer Correctness, Citation Accuracy, Abstention, Conflict Detection (see `EVALUATION.md`).

## Architecture

Pipeline: `decompose → retrieve (hybrid, per sub-query) → answer (JSON) → verify citations`.
The LLM returns a structured JSON (answer, sufficient, conflict, citations). Our code verifies that every cited passage number really exists and rejects answers without valid citations.

### Project layout

## Dataset

- **HotpotQA** (distractor, validation split), shuffled with seed 42 → **50 005 unique paragraphs** (reproducible subset), 5 421 questions with supporting facts as ground truth.
- **Synthetic incident corpus** (6 documents in md/docx/pdf/txt) with a deliberate **conflict** (47 vs 52 minutes), a **prompt-injection** document, and unanswerable questions. 14 questions.
- Exact duplicates: 18 found and removed. Near-duplicates: 62 found by MinHash; **flagged, not removed** (they were false positives: same template, different facts, and one was needed by the eval).

## Technical decisions

| Decision | Why |
|---|---|
| PostgreSQL + pgvector (Neon) | One DB for documents, chunks, vectors and metadata; SQL CRUD with `ON DELETE CASCADE`; free tier |
| `bge-small-en-v1.5` | Small (384 dims), runs on CPU, good quality; GPU used for indexing |
| Own BM25 | Postgres `ts_rank_cd` had Recall@10 0.145 vs 0.785 for our BM25 (no IDF) |
| RRF fusion | Score scales of dense and sparse are incomparable; ranks are not |
| JSON-based tool calling | One code path for Groq and Gemini fallback; full control and validation |
| One LLM call for answer+evidence+conflict | Free-tier rate limits |
| Single chunking strategy in DB | 500 MB free DB limit (291 MB used); strategies compared offline |
| Config in `.env` / secrets | Models and keys never in code |

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or cpu
pip install fastapi "uvicorn[standard]" streamlit pydantic-settings python-dotenv \
    datasets pypdf python-docx "psycopg[binary]" pgvector sentence-transformers \
    google-genai groq reportlab requests python-multipart pytest numpy tiktoken

cp .env.example .env     # fill DATABASE_URL, GROQ_API_KEY, GEMINI_API_KEY (+ models)
python scripts/download_hotpot.py
python scripts/index_corpus.py structural
python scripts/make_incident_corpus.py && python scripts/ingest_folder.py data/incidents

uvicorn app.api.main:app --port 8000      # API + /docs
streamlit run ui/app.py                   # UI
pytest                                    # tests
python eval/retrieval_eval.py
python eval/e2e_eval.py run pipeline && python eval/e2e_eval.py report
```

`.env`:

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | DB check, counts |
| `GET /documents` | list / search by title |
| `POST /documents` | upload; create / update (hash changed) / unchanged |
| `DELETE /documents/{doc_id}` | delete with cascade |
| `POST /ask` | `mode`: `pipeline` or `agent` |
| `GET /sources/{chunk_id}` | full cited chunk |

## Deployment

- **Architecture:** Hugging Face Space (free CPU, 16 GB RAM) runs the app; Neon (free PostgreSQL) stores data, so data survives restarts; secrets are stored in Space secrets, never in git.
- **Docker Space is paid** on our account, so deployment uses the **Gradio SDK**: `space_app.py` mounts a Gradio UI on the FastAPI app.
- **Status:** the Space build succeeds, but the runtime start is **not yet working** (the app starts when run locally; on the Space the container does not reach "Running"). Root cause not yet confirmed. Live demo link: _to be added_.
- **Free-tier limits:** the Space sleeps after inactivity; the first question builds the BM25 index in memory (~15-60 s); Neon free tier is 500 MB; Groq/Gemini free tiers are rate-limited and Gemini returned 503/504 frequently, hence timeouts, retries and provider fallback.

## Known limitations

- BM25 index is in memory and rebuilt after upload/delete.
- Refusal detection heuristic is English-only.
- Agent mode is less stable than the fixed pipeline (see `EVALUATION.md`).
- Conflict detection is prompt-based and only partly reliable (1 of 2 test cases).
- Multi-hop retrieval still misses the second document in ~23% of cases.
