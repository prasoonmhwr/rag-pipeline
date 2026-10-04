# Building a Production-Ready RAG Pipeline: pgvector + FastAPI

A complete, ground-up guide. By the end you'll understand *why* every piece exists, not just how to copy it.

---

## 1. Architecture: The Mental Model First

RAG (Retrieval-Augmented Generation) has two pipelines that share storage but run independently:

```
INGESTION PIPELINE (offline/batch)
  Document → Loader → Chunker → Embedder → Postgres(pgvector)

QUERY PIPELINE (online/low-latency)
  User question → Embedder → Vector search (+ optional keyword search)
                → Rerank → Assemble context → LLM → Answer
```

**Why this split matters**: ingestion is throughput-bound (you want to process thousands of documents cheaply), while querying is latency-bound (a user is waiting, typically <500ms budget for retrieval). You will tune these two paths completely differently — batching and background workers for ingestion, connection pooling and caching for queries. Conflating them in your head is the #1 reason people build RAG systems that fall over in production.

The database is the connective tissue. Postgres + pgvector is a strong choice for production (versus a dedicated vector DB) because:
- You get ACID transactions across your business data *and* vectors — no dual-write consistency problems.
- You can join vector similarity with relational filters (`WHERE tenant_id = ? AND created_at > ?`) in a single query.
- Operational maturity: backups, replication, monitoring — all the Postgres tooling you already trust.
- Trade-off you're accepting: at very large scale (100M+ vectors) a purpose-built vector DB (Qdrant, Milvus) may out-perform pgvector on raw ANN throughput. For the vast majority of real applications (under tens of millions of vectors), pgvector is not the bottleneck.

---

## 2. Environment Setup

### 2.1 Docker Compose (Postgres + pgvector)

```yaml
# docker-compose.yml
version: "3.9"
services:
  db:
    image: pgvector/pgvector:pg16
    restart: always
    environment:
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
      POSTGRES_DB: rag_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    command: >
      postgres
      -c shared_buffers=1GB
      -c max_connections=200
      -c work_mem=64MB
      -c maintenance_work_mem=512MB
volumes:
  pgdata:
```

`work_mem` and `maintenance_work_mem` matter more than people expect — index builds (HNSW/IVFFlat) are memory-hungry, and starving them causes disk spills that make index creation take hours instead of minutes on large tables.

### 2.2 Python dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.35
pgvector==0.3.4
pydantic==2.9.2
pydantic-settings==2.5.2
openai==1.51.0          # or your embedding provider of choice
tenacity==9.0.0          # retries
redis==5.0.8             # caching / rate limiting
tiktoken==0.8.0           # token-aware chunking
httpx==0.27.2
python-multipart==0.0.12  # file uploads
```

Install with `pip install -r requirements.txt`.

---

## 3. Database Schema Design (the part everyone gets subtly wrong)

### 3.1 Enable the extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3.2 Core schema

```sql
-- Source documents (the parent record)
CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       UUID NOT NULL,               -- multi-tenancy from day one
    source_uri      TEXT NOT NULL,                -- s3://..., file path, url
    title           TEXT,
    content_hash    TEXT NOT NULL,                -- sha256 of raw content, for dedup/re-ingest detection
    metadata        JSONB DEFAULT '{}'::jsonb,    -- author, tags, doc type, etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX idx_documents_tenant_hash ON documents (tenant_id, content_hash);

-- Chunks (the retrievable unit — this is what actually gets embedded)
CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id       UUID NOT NULL,                -- denormalized for query-time filtering without a join
    chunk_index     INT NOT NULL,                 -- position within the document
    content         TEXT NOT NULL,
    token_count     INT NOT NULL,
    embedding       vector(1536),                 -- match your embedding model's dimension EXACTLY
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_chunks_tenant_id ON chunks (tenant_id);

-- Full-text search support (for hybrid search later)
ALTER TABLE chunks ADD COLUMN content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN (content_tsv);
```

**Design decisions worth understanding, not memorizing:**

- **`documents` vs `chunks` as separate tables**: you almost always retrieve chunks, but you need to trace back to the source document for citations, re-ingestion, and deletion cascades. Never embed the whole document as one vector for anything beyond trivial use cases — a 10-page PDF as a single 1536-dim vector destroys retrieval precision because the vector becomes an average of unrelated concepts.
- **`content_hash` for dedup**: re-running ingestion on unchanged documents is a common source of runaway embedding API costs. Hash the source content; skip re-embedding if the hash matches.
- **`tenant_id` denormalized onto `chunks`**: if you only put it on `documents`, every query needs a join before it can filter, which prevents the query planner from using a tenant-scoped index efficiently at scale. Denormalizing this one column is a well-worn production pattern.
- **`vector(1536)`**: pgvector requires a fixed dimension per column, declared at table-creation time. 1536 is OpenAI's `text-embedding-3-small`. If you switch embedding models, dimension mismatches will throw at insert time — this is a good thing; it's your reminder to migrate.
- **Generated `tsvector` column**: computing this at write time (not query time) means your hybrid search queries don't pay a tokenization cost on every request.

### 3.3 Vector indexes: HNSW vs IVFFlat — the decision that actually matters

```sql
-- HNSW (recommended default for most production workloads)
CREATE INDEX idx_chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

```sql
-- IVFFlat (alternative — cheaper to build, costs recall)
CREATE INDEX idx_chunks_embedding_ivfflat ON chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

| | HNSW | IVFFlat |
|---|---|---|
| Build time | Slower | Faster |
| Build memory | Higher | Lower |
| Query speed | Faster, more consistent | Slower at same recall |
| Recall | Higher at same speed | Needs tuning (`lists`, `probes`) |
| Needs pre-existing data? | No | Yes — needs representative data to compute cluster centroids before building |
| Insert cost | Moderate, degrades gracefully | Needs periodic `REINDEX` as data grows past cluster assumptions |

**Rule of thumb**: use HNSW unless you have a very specific reason not to (e.g., extremely memory-constrained environment, or you need index builds to be fast because you rebuild frequently). Almost all modern pgvector production deployments default to HNSW.

**Distance operators** — pick to match your embedding model's training objective:
- `vector_cosine_ops` → cosine distance (`<=>`). Use this for OpenAI/most modern embedding models (they're normalized, so cosine ≈ dot product but numerically more stable).
- `vector_l2_ops` → Euclidean distance (`<->`).
- `vector_ip_ops` → negative inner product (`<#>`). Use only if your model is explicitly trained for dot-product similarity on **unnormalized** vectors.

Using the wrong operator doesn't error — it just silently gives you worse retrieval quality. This is the single most common invisible bug in pgvector setups.

**Tuning HNSW query-time recall/speed trade-off:**

```sql
SET hnsw.ef_search = 100;  -- higher = more accurate, slower. Default is 40.
```

Set this per-session or per-query based on your latency budget, not globally — a support-ticket search might tolerate `ef_search=200` while an autocomplete-style query needs `ef_search=40`.

---

## 4. Chunking Strategy (this determines retrieval quality more than your embedding model does)

Three approaches, in order of sophistication:

### 4.1 Fixed-size with overlap (baseline, works surprisingly well)

```python
import tiktoken

def chunk_fixed(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += chunk_size - overlap   # overlap prevents context from being severed at chunk boundaries
    return chunks
```

Why overlap matters: without it, a sentence that spans a chunk boundary gets split, and neither half carries enough context to be retrieved for a question about that sentence.

### 4.2 Recursive / structure-aware chunking (better default for real documents)

Split on semantic boundaries first (paragraphs → sentences → words), falling back only when a unit is still too large:

```python
def chunk_recursive(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]
    enc = tiktoken.get_encoding("cl100k_base")

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(enc.encode(text)) <= chunk_size:
            return [text]
        if not seps:
            # last resort: hard token split
            tokens = enc.encode(text)
            return [enc.decode(tokens[i:i+chunk_size]) for i in range(0, len(tokens), chunk_size)]

        sep, rest_seps = seps[0], seps[1:]
        parts = text.split(sep)
        results, buffer = [], ""
        for part in parts:
            candidate = buffer + sep + part if buffer else part
            if len(enc.encode(candidate)) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    results.extend(_split(buffer, rest_seps))
                buffer = part
        if buffer:
            results.extend(_split(buffer, rest_seps))
        return results

    return _split(text, separators)
```

### 4.3 Semantic chunking (highest quality, highest cost)

Embed sentences, then split where consecutive-sentence similarity drops sharply (a topic shift). This costs extra embedding calls at ingestion time but produces chunks that are internally coherent. Only worth it for high-value corpora (legal, medical) where retrieval precision directly affects outcomes — not worth the complexity for most applications.

**Practical guidance**: start with recursive chunking at 400–600 tokens with 10–15% overlap. Tune based on retrieval evaluation (Section 10), not intuition.

---

## 5. Embedding Generation Service

Wrap your embedding provider so you can swap models, batch requests, retry on failure, and cache — none of which you want scattered across your ingestion code.

```python
# app/services/embeddings.py
import asyncio
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = AsyncOpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_BATCH_SIZE = 100   # provider-imposed batch limits vary — check your provider's docs

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]

async def embed_texts(texts: list[str], concurrency: int = 5) -> list[list[float]]:
    """Batches + parallelizes embedding calls with bounded concurrency."""
    batches = [texts[i:i + MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(batch):
        async with semaphore:
            return await _embed_batch(batch)

    results = await asyncio.gather(*[_run(b) for b in batches])
    return [emb for batch_result in results for emb in batch_result]
```

Why each piece exists:
- **Retry with exponential backoff**: embedding APIs rate-limit and occasionally 5xx. Without retries, a transient blip fails your entire ingestion job.
- **Batching**: embedding 1000 chunks one-at-a-time is both slow and wasteful — providers charge per-request overhead and most support batch input natively.
- **Bounded concurrency (semaphore)**: parallelizing batches speeds up ingestion, but unbounded concurrency will get you rate-limited or blow past provider concurrency caps. Tune the `concurrency` value against your provider's actual limits.

---

## 6. The Ingestion Pipeline

```python
# app/services/ingestion.py
import hashlib
import uuid
from sqlalchemy import text
from app.db import AsyncSessionLocal
from app.services.embeddings import embed_texts
from app.services.chunking import chunk_recursive

async def ingest_document(
    tenant_id: uuid.UUID,
    source_uri: str,
    title: str,
    raw_text: str,
    metadata: dict | None = None,
) -> uuid.UUID:
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    async with AsyncSessionLocal() as session:
        # 1. Dedup check — skip expensive re-embedding if content is unchanged
        existing = await session.execute(
            text("SELECT id FROM documents WHERE tenant_id = :tid AND content_hash = :hash"),
            {"tid": tenant_id, "hash": content_hash},
        )
        row = existing.first()
        if row:
            return row[0]   # already ingested, no-op

        # 2. Insert the document record
        doc_id = uuid.uuid4()
        await session.execute(
            text("""
                INSERT INTO documents (id, tenant_id, source_uri, title, content_hash, metadata)
                VALUES (:id, :tid, :uri, :title, :hash, :meta)
            """),
            {"id": doc_id, "tid": tenant_id, "uri": source_uri, "title": title,
             "hash": content_hash, "meta": metadata or {}},
        )

        # 3. Chunk
        chunks = chunk_recursive(raw_text, chunk_size=512, overlap=64)

        # 4. Embed (batched, concurrent)
        embeddings = await embed_texts(chunks)

        # 5. Bulk insert chunks + embeddings
        chunk_rows = [
            {
                "id": uuid.uuid4(),
                "document_id": doc_id,
                "tid": tenant_id,
                "idx": i,
                "content": chunk,
                "token_count": len(chunk.split()),  # swap for tiktoken count if you need precision
                "embedding": str(emb),  # asyncpg/pgvector adapter handles list->vector; see note below
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]
        await session.execute(
            text("""
                INSERT INTO chunks (id, document_id, tenant_id, chunk_index, content, token_count, embedding)
                VALUES (:id, :document_id, :tid, :idx, :content, :token_count, :embedding)
            """),
            chunk_rows,
        )
        await session.commit()

    return doc_id
```

**Production hardening for this pipeline:**
- Run ingestion as a background job (Celery, arq, or a simple asyncio task queue), never inline on an HTTP request — embedding hundreds of chunks can take seconds to minutes, and you don't want to hold an HTTP connection open for that.
- Wrap the whole function in a try/except that marks the document as `status = 'failed'` in metadata on error, so failed ingestions are visible and retryable rather than silently lost.
- For very large documents, insert chunks in batches of ~500 rows rather than one giant `INSERT`, to avoid a single massive transaction that locks resources and complicates retry-on-partial-failure.

---

## 7. FastAPI Application Structure

```
app/
├── main.py              # app factory, lifespan, router registration
├── config.py            # pydantic-settings config
├── db.py                # engine, session factory, connection pool
├── models.py            # Pydantic request/response schemas
├── dependencies.py      # DI: get_db_session, get_current_tenant, etc.
├── routers/
│   ├── ingest.py
│   └── query.py
└── services/
    ├── embeddings.py
    ├── chunking.py
    ├── ingestion.py
    └── retrieval.py
```

### 7.1 Config

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_db"
    openai_api_key: str
    redis_url: str = "redis://localhost:6379/0"
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20

    class Config:
        env_file = ".env"

settings = Settings()
```

### 7.2 Database engine — connection pooling done right

```python
# app/db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_min_size,
    max_overflow=settings.db_pool_max_size - settings.db_pool_min_size,
    pool_pre_ping=True,       # detects dead connections before using them — critical, avoid mysterious "connection closed" errors in prod
    pool_recycle=1800,        # recycle connections every 30 min, avoids stale connections behind load balancers/proxies
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session
```

**Why connection pooling is non-negotiable**: opening a new Postgres connection costs several milliseconds of TCP + auth handshake. Under load, without pooling, connection overhead alone can dominate your query latency. `pool_pre_ping` specifically prevents a nasty class of intermittent production errors where a connection was silently dropped (e.g., by a cloud LB idle timeout) and the first query on it fails.

### 7.3 App entrypoint with lifespan management

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db import engine
from app.routers import ingest, query

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: warm the pool by issuing a trivial query
    async with engine.connect() as conn:
        await conn.execute("SELECT 1")
    yield
    # shutdown: dispose the pool cleanly
    await engine.dispose()

app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
```

---

## 8. The Retrieval Endpoint

### 8.1 Pure vector similarity search

```python
# app/services/retrieval.py
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

async def vector_search(
    session: AsyncSession,
    tenant_id: str,
    query_embedding: list[float],
    top_k: int = 10,
    ef_search: int = 100,
) -> list[dict]:
    await session.execute(text(f"SET hnsw.ef_search = {ef_search}"))
    result = await session.execute(
        text("""
            SELECT
                c.id, c.content, c.document_id, d.title, d.source_uri,
                1 - (c.embedding <=> :query_embedding) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.tenant_id = :tenant_id
            ORDER BY c.embedding <=> :query_embedding
            LIMIT :top_k
        """),
        {"query_embedding": str(query_embedding), "tenant_id": tenant_id, "top_k": top_k},
    )
    return [dict(row._mapping) for row in result]
```

Note `1 - cosine_distance = cosine_similarity` — `<=>` returns *distance*, so we flip the sign for a human-readable similarity score. **Always filter by `tenant_id` before the `ORDER BY`** — this lets the planner use the tenant index to shrink the candidate set before doing the (expensive) similarity ranking. Never rely on filtering after retrieval for multi-tenant isolation; that's both slower and a security bug waiting to happen.

### 8.2 Hybrid search: vector + full-text, fused

Pure vector search misses exact keyword/entity matches (product codes, names, acronyms) that embeddings often blur together. Hybrid search fixes this via **Reciprocal Rank Fusion (RRF)**:

```python
async def hybrid_search(
    session: AsyncSession,
    tenant_id: str,
    query_text: str,
    query_embedding: list[float],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[dict]:
    result = await session.execute(
        text("""
            WITH vector_results AS (
                SELECT id, RANK() OVER (ORDER BY embedding <=> :query_embedding) AS rank
                FROM chunks
                WHERE tenant_id = :tenant_id
                ORDER BY embedding <=> :query_embedding
                LIMIT 50
            ),
            text_results AS (
                SELECT id, RANK() OVER (ORDER BY ts_rank_cd(content_tsv, query) DESC) AS rank
                FROM chunks, to_tsquery('english', :ts_query) query
                WHERE tenant_id = :tenant_id AND content_tsv @@ query
                ORDER BY ts_rank_cd(content_tsv, query) DESC
                LIMIT 50
            )
            SELECT
                c.id, c.content, c.document_id, d.title,
                COALESCE(1.0 / (:rrf_k + v.rank), 0.0) + COALESCE(1.0 / (:rrf_k + t.rank), 0.0) AS rrf_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN vector_results v ON v.id = c.id
            LEFT JOIN text_results t ON t.id = c.id
            WHERE v.id IS NOT NULL OR t.id IS NOT NULL
            ORDER BY rrf_score DESC
            LIMIT :top_k
        """),
        {
            "query_embedding": str(query_embedding),
            "tenant_id": tenant_id,
            "ts_query": " & ".join(query_text.split()),  # simplistic; use plainto_tsquery for robustness
            "rrf_k": rrf_k,
            "top_k": top_k,
        },
    )
    return [dict(row._mapping) for row in result]
```

RRF fuses two ranked lists without needing their scores to be on comparable scales (cosine similarity and BM25-style text rank are not directly comparable numbers) — it only uses each result's *rank position*, which is why it's the standard fusion technique for hybrid retrieval.

### 8.3 The FastAPI route

```python
# app/routers/query.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.dependencies import get_db_session, get_current_tenant
from app.services.embeddings import embed_texts
from app.services.retrieval import hybrid_search

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class RetrievedChunk(BaseModel):
    content: str
    document_title: str | None
    score: float

@router.post("/", response_model=list[RetrievedChunk])
async def query_endpoint(
    req: QueryRequest,
    session=Depends(get_db_session),
    tenant_id: str = Depends(get_current_tenant),
):
    [query_embedding] = await embed_texts([req.query])
    results = await hybrid_search(session, tenant_id, req.query, query_embedding, top_k=req.top_k)
    return [
        RetrievedChunk(content=r["content"], document_title=r["title"], score=r["rrf_score"])
        for r in results
    ]
```

---

## 9. Production Hardening

### 9.1 Caching query embeddings

Identical or near-identical queries (common in support/FAQ use cases) shouldn't hit the embedding API repeatedly.

```python
import hashlib, json
import redis.asyncio as redis
from app.config import settings

redis_client = redis.from_url(settings.redis_url)

async def cached_embed_query(query: str) -> list[float]:
    key = f"emb:{hashlib.sha256(query.encode()).hexdigest()}"
    cached = await redis_client.get(key)
    if cached:
        return json.loads(cached)
    [embedding] = await embed_texts([query])
    await redis_client.set(key, json.dumps(embedding), ex=3600)
    return embedding
```

### 9.2 Rate limiting the query endpoint

Prevent a single tenant from starving embedding-API quota or database connections for everyone else:

```python
from fastapi import HTTPException

async def rate_limit(tenant_id: str, limit: int = 60, window_seconds: int = 60):
    key = f"ratelimit:{tenant_id}"
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, window_seconds)
    if count > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
```

### 9.3 Guarding against unbounded query cost

A user-supplied `top_k=100000` or a pathological full-text query can blow your latency budget. Always clamp:

```python
class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

    @field_validator("top_k")
    @classmethod
    def clamp_top_k(cls, v):
        return max(1, min(v, 50))
```

### 9.4 Observability

Log, at minimum, per query: latency breakdown (embedding time vs DB time), tenant_id, number of results, and the `ef_search` value used. Without this, you cannot debug "search feels slow sometimes" reports, because the two dominant cost centers (external embedding API latency and Postgres query time) have very different failure modes and remedies.

```python
import time, logging

logger = logging.getLogger("rag.retrieval")

async def query_endpoint(req: QueryRequest, session=Depends(get_db_session), tenant_id=Depends(get_current_tenant)):
    t0 = time.perf_counter()
    [query_embedding] = await cached_embed_query(req.query)
    t1 = time.perf_counter()
    results = await hybrid_search(session, tenant_id, req.query, query_embedding, top_k=req.top_k)
    t2 = time.perf_counter()
    logger.info(
        "query tenant=%s embed_ms=%.1f search_ms=%.1f n_results=%d",
        tenant_id, (t1 - t0) * 1000, (t2 - t1) * 1000, len(results),
    )
    return results
```

### 9.5 Index maintenance at scale

- `HNSW` indexes degrade gracefully with inserts but still benefit from an occasional `REINDEX CONCURRENTLY` after very large bulk loads.
- Run `VACUUM ANALYZE chunks;` regularly — pgvector similarity queries rely on accurate planner statistics to decide whether to use the index at all.
- If you delete/re-ingest documents frequently, watch table bloat; `chunks` rows are wide (1536 floats ≈ 6KB per row just for the vector), so dead tuples accumulate fast.

### 9.6 Handling embedding model migrations

You will eventually want to switch embedding models (better quality, cheaper, higher dimension). Because `vector(1536)` is a fixed column dimension, plan for this:
1. Add a new column `embedding_v2 vector(N)` rather than mutating in place.
2. Backfill via a background job.
3. Build the new index concurrently (`CREATE INDEX CONCURRENTLY`, and note pgvector supports this since recent versions).
4. Cut over query traffic once backfill + new index are verified.
5. Drop the old column and index only after a rollback window has passed.

Never do an in-place dimension change on a live table — there's no safe in-place path, and half-migrated data silently corrupts retrieval quality.

---

## 10. Evaluating Retrieval Quality (skipping this is why most RAG systems underperform)

Build a small labeled eval set: (query, expected relevant chunk IDs). Compute:

- **Recall@k**: of the expected relevant chunks, what fraction appear in the top-k results?
- **MRR (Mean Reciprocal Rank)**: how high does the first relevant result rank, on average?

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 0.0
    return len(top_k & relevant_ids) / len(relevant_ids)

def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0
```

Run this eval whenever you change chunk size, overlap, embedding model, or `ef_search` — these are exactly the levers that feel "reasonable either way" until you measure them. In practice, chunk size/overlap tuning alone routinely swings recall by 10–20 percentage points.

---

## 11. Common Pitfalls (the ones that cost people days of debugging)

1. **Wrong distance operator for the embedding model** — silent, no error, just worse results. Always confirm what your embedding model was trained/normalized for.
2. **Forgetting `SET hnsw.ef_search`** — default (40) may be too low for high-recall use cases; you'll get plausible-but-wrong "closest" results and won't know why.
3. **Embedding whole documents instead of chunks** — kills precision.
4. **No tenant filter before `ORDER BY`** — both a performance and a data-isolation bug.
5. **Synchronous embedding calls inside a request handler with no timeout** — one slow provider response hangs a worker; always set `timeout=` on your HTTP client to the embedding API.
6. **Not handling embedding API partial failures in batch jobs** — if batch item 47 of 200 fails, do you retry the whole job or just that item? Design for the latter.
7. **Ignoring index build time on first deploy** — building an HNSW index on millions of existing rows can take a long time and lock the table if not done with `CONCURRENTLY`.
8. **No content-hash dedup** — re-ingesting unchanged documents burns embedding API budget for zero benefit.

---

## 12. Putting It All Together — Minimal End-to-End Run

```bash
# 1. Start Postgres
docker compose up -d

# 2. Run schema migrations (the SQL from Section 3)
psql $DATABASE_URL -f schema.sql

# 3. Start the API
uvicorn app.main:app --reload

# 4. Ingest a document
curl -X POST localhost:8000/api/v1/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"source_uri": "docs/handbook.pdf", "title": "Employee Handbook", "raw_text": "..."}'

# 5. Query
curl -X POST localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the parental leave policy?", "top_k": 5}'
```

---

## 13. Grounded, Verifiable Answers

Retrieving relevant chunks does not make the final answer grounded. The LLM can still ignore the context, blend it with prior knowledge, or state something the chunks don't actually support. "Grounded and verifiable" means: every claim in the answer can be traced to a specific retrieved chunk, and you have a mechanism to check that the trace is honest — not just asserted.

### 13.1 Force citation-structured generation

Don't ask for a free-form answer and hope citations appear. Structure the prompt and the output schema so citation is mandatory, and tie each claim to a chunk ID you already have from retrieval.

```python
# app/services/generation.py
from pydantic import BaseModel
from openai import AsyncOpenAI

client = AsyncOpenAI()

class Citation(BaseModel):
    chunk_id: str
    quote: str          # short verbatim span from the chunk supporting the claim

class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    sufficient_context: bool   # model's own signal: was retrieved context enough to answer?

SYSTEM_PROMPT = """You answer ONLY using the provided context chunks. Rules:
1. Every factual sentence in your answer must be supported by at least one chunk.
2. For each claim, cite the chunk_id it came from.
3. If the context does not contain enough information to answer, set
   sufficient_context=false and give a partial or "I don't know" answer — do not
   fill gaps from outside knowledge.
4. Never state something the context contradicts.
Respond only with JSON matching the given schema."""

async def generate_grounded_answer(query: str, chunks: list[dict]) -> GroundedAnswer:
    context_block = "\n\n".join(
        f"[chunk_id={c['id']}]\n{c['content']}" for c in chunks
    )
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "grounded_answer",
            "schema": GroundedAnswer.model_json_schema(),
        }},
    )
    return GroundedAnswer.model_validate_json(resp.choices[0].message.content)
```

Why each rule exists: rule 1 and 2 make citation a structural requirement, not a courtesy the model might skip under token pressure. Rule 3 gives you an explicit **abstention signal** rather than forcing the model to always produce a confident-sounding answer even on thin context. Rule 4 addresses the specific failure mode where retrieved context is present but the model still pattern-matches to its own prior knowledge instead of what's in front of it.

### 13.2 Verify citations are real, not hallucinated

An LLM can hallucinate a `chunk_id` or attribute a claim to a chunk that doesn't actually support it. Verify mechanically, don't trust the model's self-report:

```python
def verify_citations(answer: GroundedAnswer, retrieved_chunks: dict[str, str]) -> list[str]:
    """Returns a list of problems found. Empty list = clean."""
    problems = []
    valid_ids = set(retrieved_chunks.keys())
    for c in answer.citations:
        if c.chunk_id not in valid_ids:
            problems.append(f"Cited chunk_id {c.chunk_id} was not in the retrieved set (hallucinated citation).")
            continue
        if c.quote.strip() and c.quote.strip() not in retrieved_chunks[c.chunk_id]:
            problems.append(f"Quote for {c.chunk_id} does not appear verbatim in the chunk (possible fabrication).")
    return problems
```

This is a cheap, deterministic check — no extra LLM call needed — and it catches the two most common grounding failures: citing a chunk that was never retrieved, and quoting something the chunk doesn't actually say.

### 13.3 Claim-level entailment checking (stronger, costs more)

For higher-stakes domains, go beyond citation matching and verify that each sentence in the answer is actually *entailed* by its cited chunk, not just adjacent to it. This is a second, smaller LLM call acting as a judge:

```python
VERIFY_PROMPT = """You are a fact-checker. Given a CLAIM and a SOURCE passage, respond
with exactly one word: SUPPORTED, CONTRADICTED, or NOT_ADDRESSED.
SOURCE: {source}
CLAIM: {claim}"""

async def check_entailment(claim: str, source: str) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",   # cheap model is fine for a binary/ternary judgment
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(source=source, claim=claim)}],
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip()
```

Run this per-sentence for high-stakes answers (medical, legal, financial); skip it for low-stakes conversational use where the mechanical citation check in 13.2 is enough. This is a latency/cost trade-off you make explicitly per use case, not a default-on for everything.

### 13.4 Abstention thresholds — knowing when *not* to answer

Combine two independent signals rather than trusting either alone:

```python
MIN_SIMILARITY_THRESHOLD = 0.72   # tune against your eval set — this is domain-specific, not universal

async def answer_with_abstention(query: str, session, tenant_id: str) -> dict:
    [query_embedding] = await cached_embed_query(query)
    results = await hybrid_search(session, tenant_id, query, query_embedding, top_k=8)

    top_similarity = results[0]["rrf_score"] if results else 0.0
    if not results or top_similarity < MIN_SIMILARITY_THRESHOLD:
        return {"answer": "I don't have enough information in the knowledge base to answer that.",
                "citations": [], "abstained": True, "reason": "low_retrieval_confidence"}

    chunk_map = {r["id"]: r["content"] for r in results}
    answer = await generate_grounded_answer(query, results)
    problems = verify_citations(answer, chunk_map)

    if not answer.sufficient_context or problems:
        return {"answer": "I found related information but can't confidently answer this.",
                "citations": [], "abstained": True, "reason": "insufficient_context_or_bad_citation",
                "debug_problems": problems}

    return {"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations], "abstained": False}
```

Two independent gates — retrieval confidence *before* generation, and citation/self-reported sufficiency *after* generation — catch different failure modes. Retrieval confidence catches "nothing relevant exists in the corpus." Post-generation checks catch "relevant chunks existed but the model still overreached or hallucinated a citation." Neither alone is sufficient; production-grade grounding needs both.

---

## 14. Confidential Data & Access Rights

This is the section most RAG tutorials skip entirely, and it's the one that causes real incidents. The core danger: **tenant-level isolation is not the same as user-level or document-level authorization.** A tenant might be a whole company, but inside that company, HR documents shouldn't be retrievable by every employee's queries.

### 14.1 The specific leakage risk unique to RAG

Even if you filter results *after* retrieval to remove chunks the user can't see, the vector index still ranked those forbidden chunks alongside allowed ones — meaning a forbidden chunk can occupy a "slot" in the top-k, silently pushing out a chunk the user *should* have seen, and in some designs its content still ends up concatenated into a prompt sent to an LLM before filtering catches it. **Filtering must happen inside the SQL `WHERE` clause, before `ORDER BY`/`LIMIT`, never as a post-processing step on already-fetched rows.**

### 14.2 ACL schema

```sql
-- Access groups (roles, teams, "confidential-hr", etc.)
CREATE TABLE access_groups (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   UUID NOT NULL,
    name        TEXT NOT NULL
);

CREATE TABLE user_group_memberships (
    user_id     UUID NOT NULL,
    group_id    UUID NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, group_id)
);

-- Which groups can see which documents. Document-level, not chunk-level —
-- chunks inherit their parent document's ACL, which keeps this table small.
CREATE TABLE document_acl (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    group_id    UUID NOT NULL REFERENCES access_groups(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, group_id)
);

CREATE INDEX idx_document_acl_group ON document_acl (group_id);
```

Document-level ACL (not per-chunk) is the right granularity for almost all cases — permissions are usually set on the source document ("who can see this HR policy PDF"), and chunks simply inherit it. Per-chunk ACLs are only worth the complexity if a single document genuinely mixes restricted and unrestricted content that must be split at the chunk level.

### 14.3 Enforce it at the database layer with Row-Level Security

Application-code filtering (`WHERE document_id IN (...)` assembled in Python) is fragile — one forgotten filter in one code path is a breach. Postgres Row-Level Security makes the filter unbypassable at the database layer itself:

```sql
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY chunk_access_policy ON chunks
    USING (
        document_id IN (
            SELECT da.document_id
            FROM document_acl da
            JOIN user_group_memberships ugm ON ugm.group_id = da.group_id
            WHERE ugm.user_id = current_setting('app.current_user_id')::uuid
        )
    );
```

Set the session variable per request, scoped to that connection's lifetime:

```python
# app/dependencies.py
async def get_db_session_for_user(user_id: str = Depends(get_current_user_id)):
    async with AsyncSessionLocal() as session:
        await session.execute(text("SET app.current_user_id = :uid"), {"uid": user_id})
        yield session
```

With RLS enabled, **every** query against `chunks` — including the vector search in Section 8 — is automatically restricted to rows the current user is allowed to see, before similarity ranking runs. This closes the leakage risk from 14.1 structurally: even if a developer forgets an ACL check in application code, the database itself won't return forbidden rows.

**Trade-off to know**: RLS adds a join/subquery cost to every query. For large ACL tables, index `user_group_memberships(user_id)` and `document_acl(group_id)` (done above), and periodically check `EXPLAIN ANALYZE` on your retrieval query to confirm the planner is using them efficiently rather than materializing the whole ACL check per row.

### 14.4 PII handling before embedding

Sensitive fields (SSNs, raw emails, medical record numbers) embedded verbatim end up baked into vector representations you can't easily "un-embed." Redact or tokenize before chunking, not after:

```python
import re

PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}

def redact_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{label.upper()}]", text)
    return text
```

This is necessarily incomplete (regex won't catch everything) — for regulated domains, use a dedicated PII-detection model/service rather than hand-rolled patterns, and treat this as defense-in-depth alongside access controls, not a replacement for them.

### 14.5 Encryption and audit logging

- Encrypt the database at rest (most managed Postgres providers do this by default; confirm it's on).
- For especially sensitive `content` fields, consider application-level column encryption in addition to at-rest disk encryption, so a database backup leak doesn't expose plaintext.
- Log every query with `user_id`, `tenant_id`, timestamp, and which `document_id`s were returned. This audit trail is what lets you answer "who accessed this confidential document, and when" after the fact — a requirement in most compliance frameworks (SOC 2, HIPAA), and the only way to detect a permissions bug after the fact rather than never knowing it happened.

---

## 15. Scaling in Production

### 15.1 What actually breaks first, in order

1. **HNSW build memory/time** — building the index on a large existing table is the first wall most teams hit, typically in the tens-of-millions-of-rows range depending on hardware. Symptom: index creation takes hours or OOMs.
2. **Connection pool exhaustion** — under real concurrent load, the FastAPI-side pool (Section 7.2) is not the actual bottleneck; Postgres's own `max_connections` is, especially if you run multiple API instances each with their own pool.
3. **Write contention on `chunks` during bulk ingestion** — large ingestion jobs competing with live query traffic for the same table/index.
4. **Single-primary write bottleneck** — once ingestion volume itself is heavy (many tenants ingesting continuously), the primary's write throughput becomes the ceiling.

### 15.2 Connection pooling at the infrastructure level: PgBouncer

Once you have more than one API instance, each maintaining its own SQLAlchemy pool, you can exceed Postgres's `max_connections` even though each app-level pool looks reasonably sized. Put PgBouncer (or your cloud provider's equivalent, e.g., RDS Proxy) between the app and Postgres, in `transaction` pooling mode:

```ini
# pgbouncer.ini
[databases]
rag_db = host=postgres-primary port=5432 dbname=rag_db

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

This lets you scale API instances horizontally without each one linearly adding to Postgres's real connection count.

### 15.3 Partitioning `chunks` for very large corpora

For multi-tenant systems with uneven tenant sizes, declarative partitioning by `tenant_id` (hash) or by ingestion time (range) keeps individual indexes smaller and lets you drop old partitions cheaply instead of running expensive `DELETE`s:

```sql
CREATE TABLE chunks (
    id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    -- ... other columns ...
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
) PARTITION BY HASH (tenant_id);

CREATE TABLE chunks_p0 PARTITION OF chunks FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE chunks_p1 PARTITION OF chunks FOR VALUES WITH (MODULUS 8, REMAINDER 1);
-- ... through p7
```

Each partition gets its own HNSW index, which means index builds and vacuum operations become parallelizable and individually smaller — a single giant HNSW graph on a 200M-row unpartitioned table is a much worse operational shape than eight 25M-row graphs. Only reach for this once you've actually measured that a single index is your bottleneck (Section 3.3's guidance still holds for smaller deployments); partitioning adds real operational complexity (cross-partition queries, rebalancing) that isn't worth paying for prematurely.

### 15.4 Read replicas for query traffic

Route the query pipeline (Section 8) to read replicas and keep the ingestion pipeline (Section 6) writing only to the primary:

```python
write_engine = create_async_engine(settings.primary_database_url)
read_engine = create_async_engine(settings.replica_database_url)

ReadSessionLocal = async_sessionmaker(read_engine, expire_on_commit=False)
WriteSessionLocal = async_sessionmaker(write_engine, expire_on_commit=False)
```

Watch for **replication lag**: a chunk just ingested on the primary may not be immediately queryable on a replica. For most knowledge-base use cases (documents added continuously, not read back within milliseconds of being written) this is a non-issue; for use cases requiring immediate read-after-write consistency, route that specific request to the primary instead.

### 15.5 Async ingestion at scale: queue-based workers

Once ingestion volume outgrows "a background asyncio task is enough," move to a real queue (Redis-backed `arq`, or Celery/RabbitMQ) so you get backpressure, retries, and horizontal worker scaling independent of the API process:

```python
# worker.py (using arq)
from arq import create_pool
from arq.connections import RedisSettings

async def ingest_task(ctx, tenant_id, source_uri, title, raw_text, metadata):
    await ingest_document(tenant_id, source_uri, title, raw_text, metadata)

class WorkerSettings:
    functions = [ingest_task]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10          # bounds concurrent ingestion jobs per worker process
```

The API route just enqueues and returns immediately:

```python
@router.post("/")
async def ingest_endpoint(req: IngestRequest, redis_pool=Depends(get_arq_pool)):
    job = await redis_pool.enqueue_job("ingest_task", req.tenant_id, req.source_uri, req.title, req.raw_text, req.metadata)
    return {"job_id": job.job_id, "status": "queued"}
```

This decouples ingestion throughput from API request/response latency entirely, and lets you scale worker count based on queue depth rather than API traffic.

### 15.6 When pgvector genuinely isn't enough anymore

If you're past several hundred million vectors with strict single-digit-millisecond latency requirements at high query throughput, or need features like GPU-accelerated ANN search, it's reasonable to evaluate a dedicated vector database. This is a real ceiling, but it's much higher than most teams assume — measure your actual bottleneck (Section 15.1's list) before concluding you've hit it.

---

## 16. Continuous Retrieval Quality Improvement

Section 10 gave you one-off Recall@k/MRR metrics. Production quality improvement is about turning that into an ongoing loop, plus techniques that improve quality beyond what chunking/embedding tuning alone can achieve.

### 16.1 Two-stage retrieval with reranking

Vector/hybrid search (Section 8) is optimized for speed over a large candidate pool, which caps its precision. Add a second, more expensive but more accurate reranking stage over a small candidate set:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    pairs = [(query, c["content"]) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
```

Pattern: retrieve 50 candidates cheaply via hybrid search, rerank down to the top 5–10 with the cross-encoder, and only those go into the LLM prompt. Cross-encoders jointly attend to the query and passage together (unlike embedding similarity, which compares independently-computed vectors), which is why they're consistently more accurate at final ranking — at the cost of not being usable over the full corpus, only over an already-small candidate set.

### 16.2 Query rewriting and expansion

User queries are often underspecified relative to how the source documents are phrased. Rewrite before embedding:

```python
REWRITE_PROMPT = """Rewrite the user's question to be more specific and to include
likely synonyms or related terms that might appear in a knowledge base. Return only
the rewritten query, no explanation.
Question: {query}"""

async def rewrite_query(query: str) -> str:
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
    )
    return resp.choices[0].message.content.strip()
```

A related, often more powerful technique is **HyDE (Hypothetical Document Embeddings)**: ask the LLM to write a hypothetical *answer* to the question, then embed that hypothetical answer instead of the raw question. Answers tend to be phrased more like the source documents than questions are, which often improves cosine similarity matches — at the cost of one extra LLM call per query, so weigh it against your latency budget.

### 16.3 Turning production usage into an improvement loop

```sql
CREATE TABLE query_feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text      TEXT NOT NULL,
    retrieved_chunk_ids UUID[] NOT NULL,
    clicked_chunk_id UUID,             -- which result, if any, the user actually engaged with
    thumbs          SMALLINT,          -- 1 = up, -1 = down, NULL = no explicit feedback
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- **Implicit signal**: if users consistently click result #3 instead of #1, that's evidence your ranking is systematically off for a class of queries — mine this for chunking/reranking tuning.
- **Explicit signal**: thumbs down on an answer should route the query + retrieved chunks into a review queue. Confirmed retrieval failures get added to your eval set (Section 10) as new labeled examples, so your eval set grows from real production failures over time instead of staying static.
- **Never auto-tune ranking weights directly from raw feedback in real time** — feedback is noisy and biased toward whatever your current ranking already surfaces (position bias: people click top results more regardless of true relevance). Use it to build better *offline* eval sets and guide deliberate changes, not as an online reinforcement signal without safeguards.

### 16.4 Regression-test retrieval changes in CI

Treat your eval set (Section 10) as a test suite, not a one-time report:

```python
# tests/test_retrieval_quality.py
import pytest
from app.services.retrieval import hybrid_search
from eval_dataset import EVAL_CASES   # list of (query, expected_relevant_chunk_ids)

MIN_ACCEPTABLE_RECALL_AT_5 = 0.75

@pytest.mark.asyncio
async def test_retrieval_recall_regression(session):
    scores = []
    for query, expected_ids in EVAL_CASES:
        results = await hybrid_search(session, TEST_TENANT_ID, query, await embed(query), top_k=5)
        retrieved_ids = [r["id"] for r in results]
        scores.append(recall_at_k(retrieved_ids, set(expected_ids), k=5))
    avg_recall = sum(scores) / len(scores)
    assert avg_recall >= MIN_ACCEPTABLE_RECALL_AT_5, f"Retrieval quality regressed: {avg_recall:.2f}"
```

Run this on every PR that touches chunking, embedding model choice, or ranking logic. This is what turns "we think this change improved retrieval" into a verified fact, and it's the single highest-leverage practice for avoiding silent quality regressions as the system evolves.

### 16.5 A/B testing configuration changes

For changes too risky to ship globally on eval-set confidence alone (new embedding model, new chunk size), route a percentage of live traffic to the new config and compare feedback-derived metrics (Section 16.3) between arms before full rollout — the same way you'd A/B test any other product change.

---

## 17. Maintaining Knowledge Sources Over Time

A RAG system's quality decays continuously if the underlying knowledge isn't actively maintained — stale documents get retrieved and cited as if current, deleted source material lingers in the vector store, and nobody notices until a user acts on outdated information.

### 17.1 Track source freshness explicitly

```sql
ALTER TABLE documents ADD COLUMN last_verified_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN source_updated_at TIMESTAMPTZ;   -- from the source system, not ingestion time
ALTER TABLE documents ADD COLUMN staleness_ttl_days INT DEFAULT 90;
ALTER TABLE documents ADD COLUMN status TEXT DEFAULT 'active';    -- active | stale | superseded | deleted
```

```sql
-- Find documents overdue for re-verification
SELECT id, title, last_verified_at
FROM documents
WHERE status = 'active'
  AND last_verified_at < now() - (staleness_ttl_days || ' days')::interval;
```

Surface staleness to the end user, not just internally — a citation like "Employee Handbook, last verified 14 months ago" lets the user judge trust for themselves, which is a meaningfully different (and more honest) experience than presenting all retrieved content as equally current.

### 17.2 Scheduled re-sync jobs

For sources with an upstream system of record (Confluence, a CMS, a shared drive), run periodic sync jobs rather than one-time ingestion:

```python
async def resync_source(source_connector, tenant_id: str):
    remote_docs = await source_connector.list_documents()
    for remote_doc in remote_docs:
        content_hash = hashlib.sha256(remote_doc.content.encode()).hexdigest()
        existing = await get_document_by_source_uri(tenant_id, remote_doc.uri)

        if existing is None:
            await ingest_document(tenant_id, remote_doc.uri, remote_doc.title, remote_doc.content)
        elif existing.content_hash != content_hash:
            await reingest_changed_document(existing.id, remote_doc.content)   # see 17.3
        else:
            await mark_verified(existing.id)   # unchanged — just bump last_verified_at, no re-embedding cost
```

The dedup check from Section 6 is what makes this cheap to run frequently: unchanged documents cost a single hash comparison, not a re-embedding pass.

### 17.3 Handling updates without full re-ingestion cost

Re-embedding an entire large document because one paragraph changed wastes API cost and briefly invalidates good chunks. Diff at the chunk level:

```python
async def reingest_changed_document(document_id: uuid.UUID, new_raw_text: str):
    new_chunks = chunk_recursive(new_raw_text)
    new_hashes = {hashlib.sha256(c.encode()).hexdigest(): c for c in new_chunks}

    existing_chunks = await get_chunks_for_document(document_id)
    existing_hashes = {hashlib.sha256(c.content.encode()).hexdigest(): c for c in existing_chunks}

    to_delete = [c for h, c in existing_hashes.items() if h not in new_hashes]
    to_add_text = [text for h, text in new_hashes.items() if h not in existing_hashes]
    # chunks whose hash matches exactly need nothing — they carry over untouched, embedding included

    if to_delete:
        await delete_chunks([c.id for c in to_delete])
    if to_add_text:
        new_embeddings = await embed_texts(to_add_text)
        await insert_chunks(document_id, to_add_text, new_embeddings)

    await touch_document_updated_at(document_id)
```

This turns "the document changed" into "only the genuinely changed chunks get re-embedded," which matters enormously for large, frequently-edited documents (wikis, policy docs) where most content is stable between edits.

### 17.4 Deletion: tombstone, don't just drop

```sql
UPDATE documents SET status = 'deleted', updated_at = now() WHERE id = :doc_id;
-- chunks remain queryable by ID for audit purposes but are excluded from retrieval:
```

```sql
-- retrieval queries must always exclude non-active documents
WHERE c.tenant_id = :tenant_id
  AND d.status = 'active'
ORDER BY c.embedding <=> :query_embedding
```

Hard-deleting immediately loses your ability to audit "why did the system once cite this" or to restore accidentally-removed content. Run actual `DELETE`s (and vector/index space reclamation) as a separate, delayed cleanup job — e.g., purge tombstoned rows older than 90 days — rather than coupling it to the user-facing delete action.

### 17.5 Superseding documents

When a new policy version replaces an old one, don't just delete the old one silently — link them, so historical questions ("what was the policy last year") remain answerable while current queries only surface the active version:

```sql
ALTER TABLE documents ADD COLUMN supersedes_document_id UUID REFERENCES documents(id);
```

Set the old document's `status = 'superseded'` and point the new one's `supersedes_document_id` at it. Filter `status = 'active'` for normal retrieval; allow an explicit "include historical versions" query mode for audit or comparison use cases.

### 17.6 Monitoring the health of your source pipeline itself

Track, per source connector: last successful sync time, consecutive failure count, and document count drift (a sudden 40% drop in documents from a source is far more likely a broken sync than a real deletion event). Alert on sync failures the same way you'd alert on any other production job failure — a silently broken ingestion job is functionally identical to slowly deleting your knowledge base, and it's the hardest kind of RAG failure to notice because retrieval still "works," it's just working over an increasingly stale or incomplete corpus.

---

## What to Practice Until It's Automatic

To genuinely internalize this, drill these until you can do them without reference:
1. Write the `CREATE TABLE` + HNSW index from memory, including why each column exists.
2. Explain, out loud, why `tenant_id` is denormalized onto `chunks`.
3. Write the RRF hybrid search query by hand.
4. Explain the difference between HNSW and IVFFlat, and when you'd choose each.
5. Trace a query request end-to-end: embed → cache check → SQL → RRF fusion → response — and name the failure mode at each step.
6. Explain why post-retrieval filtering for access control is unsafe, and write the RLS policy that fixes it from memory.
7. Write out the two independent gates (pre-generation retrieval confidence, post-generation citation verification) that together produce a defensible abstention decision.
8. Explain why cross-encoder reranking outperforms embedding similarity alone, and why it can't replace vector search over the full corpus.
9. Walk through what happens, end to end, when a source document is edited: which chunks get re-embedded, which are untouched, and why that distinction is cheap to compute (content-hash comparison) rather than expensive to guess.
10. Name the four things that break first as a pgvector deployment scales, in order, and the concrete fix for each.