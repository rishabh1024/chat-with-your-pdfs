# Chat With PDFs - FastAPI Backend

A FastAPI backend for **document-grounded AI chat**. Upload PDFs, index them into **MongoDB Atlas Vector Search**, and talk to a **LangGraph RAG agent** that retrieves relevant chunks to answer questions — with auth, conversation history, and real-time indexing status.

---

## What it does

| Capability | How it works |
|---|---|
| **PDF upload** | Authenticated users upload PDFs to Supabase Storage (deduped by SHA-256). |
| **Vector indexing** | Background job chunks the PDF, enriches metadata with an LLM, embeds via OpenRouter, and stores vectors in MongoDB Atlas. |
| **Live indexing status** | Clients subscribe to Server-Sent Events (`/upload/index_status/{document_id}/events`) until indexing finishes. |
| **RAG conversations** | A LangGraph agent searches the vector store with a tool call, then answers. History is checkpointed in Postgres. |
| **Secure access** | Every route requires a Supabase JWT (`Authorization: Bearer …`). |

**Typical flow**

```
Upload PDF → Store in Supabase → Chunk + embed → MongoDB Atlas Vector Search
                                                      ↓
Create conversation → Send message → Agent retrieves chunks → AI reply
```

---

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, uvicorn |
| Auth | Supabase Auth (JWKS / JWT) |
| Relational data | PostgreSQL (Supabase) + SQLAlchemy async + asyncpg |
| Agent memory | LangGraph `PostgresSaver` |
| Object storage | Supabase Storage |
| Vectors | MongoDB Atlas Vector Search + LangChain MongoDB |
| LLMs / embeddings | OpenRouter (`langchain-openrouter`, OpenAI-compatible embeddings) |
| Tooling | `uv`, Ruff, Pyright, Pytest |

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI (app.py)                        │
│  CORS · exception handlers · lifespan (DB, agent, storage)      │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
        /conversations/*              /upload/*
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│ ConversationService       │   │ FileUploadService                 │
│  · CRUD + chat            │   │  · validate PDF                   │
│  · Message persistence    │   │  · hash + store (Supabase)        │
│  · Ownership checks       │   │  · schedule background indexing   │
└─────────────┬─────────────┘   └─────────────────┬───────────────┘
              │                                   │
              ▼                                   ▼
┌───────────────────────────┐   ┌─────────────────────────────────┐
│ RAGAgent (LangGraph)      │   │ DocumentIndexer                   │
│  · OpenRouter chat models │   │  load → clean → chunk → metadata  │
│  · search_documents tool  │   │  → embeddings → Atlas collection  │
│  · Postgres checkpointer  │   └─────────────────┬───────────────┘
└─────────────┬─────────────┘                     │
              │                                   ▼
              │                     MongoDB Atlas (pdf_embeddings)
              ▼                     + document_index_status
     Postgres (conversations,
     messages, documents,
     LangGraph checkpoints)
```

---

### External setup checklist

Before running locally, make sure these exist:

1. **Supabase Auth** — users can sign in; you can obtain a JWT for API calls.
2. **Supabase Storage** — private bucket named `pdf-file-storage` (PDF MIME types).
3. **Supabase Postgres** — tables for conversations, messages, and documents (see models under `src/database/models.py`). LangGraph will create its checkpoint tables on startup via `PostgresSaver.setup()`.
4. **MongoDB Atlas** — database (default `sample_mflix`), collection `pdf_embeddings`, and a vector search index named `document_embeddings` on the `embedding` field (cosine similarity; dimensions must match the embedding model).

---

## Setup

```powershell
# Clone and enter the project
cd mongo-vector-backend

# Install dependencies (includes optional dev tools: pytest, ruff, pyright)
uv sync --extra dev

# Configure environment
Copy-Item .env.example .env
# Edit .env with your MongoDB, Supabase, and OpenRouter credentials
```

### Run the API

```powershell
# From the project root (loads app:app from app.py)
uv run fastapi dev app.py
```

Or with uvicorn directly:

```powershell
uv run uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Conversations — `/conversations`

| Method | Path | Description |
|---|---|---|
| `POST` | `/conversations` | Create a conversation (`{ "title": "…" }`) |
| `GET` | `/conversations` | List the current user’s conversations |
| `DELETE` | `/conversations/{conversation_id}` | Delete a conversation (owner only) |
| `POST` | `/conversations/{conversation_id}/messages` | Send a user message; returns the AI reply |
| `GET` | `/conversations/{conversation_id}/messages` | Load message history |

**Send message body**

```json
{
  "message_content": "Summarize my uploaded resume",
  "ai_model": {
    "model_name": "qwen/qwen3.8-27b:free",
    "temperature": 0.3,
    "top_p": 0.9,
    "frequency_penalty": 0.2,
    "max_tokens": 10000,
    "seed": 5
  }
}
```

### File upload — `/upload`

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload/` | Upload a PDF (`multipart/form-data`, field `input_file`) |
| `GET` | `/upload/index_status/{document_id}/events` | SSE stream until indexing completes |

**Upload response (shape)**

```json
{
  "document_id": "<sha256>",
  "file_hash": "<sha256>",
  "upload_status": "Success",
  "upload_error": null,
  "document_indexing_status": {
    "status": "Pending",
    "message": "… Connect to /index_status/{id}/events …"
  }
}
```

`document_id` is the file’s SHA-256 hash — used for storage path, deduplication, and indexing status.

---

## How the RAG pipeline works

1. **Load** — PDF pages via PyMuPDF / LangChain loaders
2. **Clean** — Drop sparse pages (< 10 words)
3. **Chunk** — Recursive character split (`chunk_size=800`, `overlap=150`)
4. **Enrich** — Structured LLM metadata (`title`, `keywords`, `hasCode`)
5. **Embed** — `qwen/qwen3-embedding-8b` through OpenRouter
6. **Store** — MongoDB Atlas collection `pdf_embeddings`, index `document_embeddings`
7. **Retrieve** — Similarity search with score threshold (`0.10`) when the agent calls `search_documents`

Conversation turns are persisted in Postgres; LangGraph checkpoints agent state per `conversation_id` so multi-turn chat keeps tool/memory context.

---

## Common development tasks

```powershell
# Install / refresh deps
uv sync --extra dev

# Run the API with hot reload
uv run fastapi dev app.py

# Run tests
uv run pytest

# Lint
uv run ruff check src tests

# Auto-fix lint issues
uv run ruff check --fix src tests

# Type-check
uv run pyright

# Add a dependency
uv add some-package

# Add a dev-only dependency
uv add --dev some-package
```

### Quick smoke test (PowerShell)

```powershell
$token = "<your-supabase-jwt>"
$headers = @{ Authorization = "Bearer $token" }

# Create a conversation
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/conversations" `
  -Headers $headers -ContentType "application/json" `
  -Body '{"title":"Portfolio demo"}'

# Upload a PDF
$form = @{ input_file = Get-Item .\sample.pdf }
Invoke-RestMethod -Method POST -Uri "http://127.0.0.1:8000/upload/" `
  -Headers $headers -Form $form
```

---

## Design notes worth highlighting

- **Auth at the edge** — Bearer JWT validated against Supabase JWKS (`ES256` / `RS256` / `HS256`); invalid or expired tokens never leak raw token material into logs.
- **Ownership** — Conversation routes resolve the resource and verify the authenticated user owns it before mutating or reading.
- **Idempotent uploads** — Same PDF bytes → same hash → storage skip + indexing skip when already successful.
- **Resilience** — Retries on embedding / metadata / Mongo timeouts; typed domain errors mapped to HTTP status codes.
- **Observability** — Structured event-style log messages (`conversation.chat.completed`, `upload.indexing.failed`, …) plus optional LangSmith tracing.

---

## Project status

Active development. Core upload → index → chat path is in place. Schema migrations (e.g. Alembic) and multi-tenant vector filtering are natural next steps if you extend this beyond a personal / demo deployment.

---

## License

Private / portfolio project unless otherwise stated. Contact the author for reuse.
