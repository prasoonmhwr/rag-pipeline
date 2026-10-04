# RAG Pipeline Architecture

Architecture diagram for the production RAG pipeline (pgvector + FastAPI + Gemini + Celery). Renders natively on GitHub — paste the block below directly into your repo's `README.md`.

```mermaid
flowchart TD
    Client["Client App<br/><i>curl / frontend / bot</i>"]

    subgraph API["API Layer"]
        FastAPI["FastAPI<br/><i>uvicorn</i>"]
        IngestRoute["/api/v1/ingest<br/><i>FastAPI router</i>"]
        QueryRoute["/api/v1/query<br/><i>FastAPI router</i>"]
        RateLimit["Rate Limiter<br/><i>redis-py</i>"]
    end

    subgraph Services["Service Layer"]
        Chunker["Chunker<br/><i>tiktoken</i>"]
        Embedder["Embedding Service<br/><i>google-genai SDK + tenacity retries</i>"]
        Cache["Embedding Cache<br/><i>redis</i>"]
        Retriever["Hybrid Retriever<br/><i>SQLAlchemy async + asyncpg</i>"]
        Reranker["Cross-Encoder Reranker<br/><i>sentence-transformers (planned)</i>"]
        Generator["Grounded Generator<br/><i>google-genai SDK (structured output)</i>"]
        Verifier["Citation Verifier<br/><i>pure Python, no extra LLM call</i>"]
    end

    subgraph Async["Background / Async Layer"]
        Queue["Job Queue<br/><i>Celery (Redis-backed)</i>"]
        Worker["Ingestion Worker Pool<br/><i>celery worker</i>"]
        Syncer["Scheduled Re-sync Job<br/><i>celery beat (placeholder task)</i>"]
        SourceConn["Source Connectors<br/><i>Confluence / S3 / CMS clients — not yet built</i>"]
    end

    subgraph Data["Data Layer"]
        PgBouncer["PgBouncer<br/><i>connection pooling — planned</i>"]
        Primary[("Postgres Primary<br/><i>pgvector + RLS</i>")]
        Replica[("Postgres Read Replica<br/><i>pgvector — planned</i>")]
        Redis[("Redis<br/><i>cache / rate limit / queue broker</i>")]
    end

    subgraph External["External Services"]
        EmbedAPI["Embedding API<br/><i>Gemini gemini-embedding-2</i>"]
        LLM["LLM API<br/><i>Gemini gemini-3.8-flash</i>"]
    end

    subgraph Ops["Cross-Cutting: Ops & Quality"]
        EvalCI["Retrieval Eval Suite<br/><i>pytest, Recall@k / MRR — planned</i>"]
        Feedback[("query_feedback table<br/><i>Postgres — planned</i>")]
        Logging["Structured Logging<br/><i>python logging</i>"]
    end

    Client -->|"POST /ingest"| IngestRoute
    Client -->|"POST /query"| QueryRoute

    IngestRoute --> RateLimit
    QueryRoute --> RateLimit
    RateLimit --> Redis

    IngestRoute -->|"enqueue job"| Queue
    Queue --> Redis
    Worker -->|"pull job"| Queue
    Worker --> Chunker
    Chunker --> Embedder
    Embedder -->|"batched calls"| EmbedAPI
    Worker -->|"bulk insert chunks + vectors"| PgBouncer

    Syncer --> SourceConn
    SourceConn -->|"content hash diff"| Worker

    QueryRoute --> Cache
    Cache --> Redis
    Cache -->|"cache miss"| Embedder
    QueryRoute --> Retriever
    Retriever -->|"vector <=> + tsvector, RLS-scoped"| PgBouncer
    Retriever --> Reranker
    Reranker --> Generator
    Generator -->|"structured JSON w/ citations"| LLM
    Generator --> Verifier
    Verifier -->|"abstain if unverified"| QueryRoute
    QueryRoute -->|"grounded answer + citations"| Client

    PgBouncer --> Primary
    PgBouncer -.->|"read-only queries"| Replica
    Primary -.->|"replication"| Replica

    QueryRoute -.->|"log clicks / thumbs"| Feedback
    Feedback --> EvalCI
    EvalCI -.->|"regression gate in CI"| Retriever

    FastAPI --- IngestRoute
    FastAPI --- QueryRoute
    QueryRoute -.-> Logging
    IngestRoute -.-> Logging
```

## Component → Tool Reference

**Built and running** (reflects the actual repo — see the main guide's table of contents for the matching section):

| Component | Tool / Library |
|---|---|
| API framework | FastAPI + Uvicorn |
| DB driver / ORM | SQLAlchemy (async) + asyncpg |
| Vector storage & search | Postgres + `pgvector` (HNSW index, cosine similarity) |
| Full-text search | Postgres `tsvector` / GIN index |
| Hybrid search fusion | Reciprocal Rank Fusion (RRF) — pure SQL |
| Access control enforcement | Postgres Row-Level Security (RLS), scoped `FOR SELECT` |
| Chunking | `tiktoken` (token-aware recursive split) |
| Embeddings | Gemini `gemini-embedding-2` via `google-genai` SDK |
| Retry logic | `tenacity` (exponential backoff) |
| Caching / rate limiting / queue broker | Redis (`redis-py`) |
| Background jobs | Celery (Redis-backed broker + result backend) |
| Grounded generation | Gemini `gemini-3.8-flash`, structured JSON output (`response_schema`) |
| Citation verification | Pure Python string matching, whitespace-normalized (no LLM call) |
| Abstention gate | Calibrated cosine-similarity threshold on top-ranked chunk |
| Observability | Python `logging` (per-query latency + citation verification results) |

**Described in the written guide, not yet built** (honest gap, not an oversight — see the guide's own "what's next" sections):

| Component | Tool / Library |
|---|---|
| Reranking | `sentence-transformers` CrossEncoder |
| Connection pooling (infra) | PgBouncer |
| Read scaling | Postgres read replica |
| Query rewriting / HyDE | An additional Gemini call to rewrite the query before embedding |
| Claim entailment check (high-stakes only) | A second Gemini call acting as a judge |
| Continuous quality gate | `pytest` + eval dataset (Recall@k, MRR) |
| Source sync connectors | Confluence / S3 / CMS clients feeding the Celery Beat placeholder task |
| Feedback capture | A `query_feedback` table for click/thumbs signal |

## Data Flow Summary

**Built and running:**
1. **Ingest path**: `Client → FastAPI → Celery Job Queue (Redis) → Worker → Chunker → Embedder (Gemini) → Postgres`
2. **Query path**: `Client → FastAPI → Hybrid Retriever (Postgres, RLS-scoped, RRF fusion) → Generator (Gemini, structured output) → Citation Verifier → abstain or answer → Client`
3. **Maintenance heartbeat**: `Celery Beat → placeholder resync task → logs that it fired` (no real source connector wired in yet)

**Planned, described in the written guide:**
4. **Full maintenance loop**: `Scheduled Syncer → Source Connectors → content-hash diff → Worker (re-embeds only changed chunks)`
5. **Quality loop**: `Query traffic → Feedback table → Eval suite → CI regression gate → back into Retriever tuning`