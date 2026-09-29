import hashlib
import json
import uuid
from sqlalchemy import text
from app.db import AsyncSessionLocal
from app.services.embeddings import embed_texts
from app.services.chunking import chunk_recursive

async def ingest_document(tenant_id: uuid.UUID, source_uri: str, title: str,
                           raw_text: str, metadata: dict | None = None) -> uuid.UUID:
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            text("SELECT id FROM documents WHERE tenant_id = :tid AND content_hash = :hash"),
            {"tid": tenant_id, "hash": content_hash},
        )
        row = existing.first()
        if row:
            return row[0]

        doc_id = uuid.uuid4()
        await session.execute(
            text("""INSERT INTO documents (id, tenant_id, source_uri, title, content_hash, metadata)
                     VALUES (:id, :tid, :uri, :title, :hash, CAST(:meta AS jsonb))"""),
            {"id": doc_id, "tid": tenant_id, "uri": source_uri, "title": title,
             "hash": content_hash, "meta": json.dumps(metadata or {})},
        )

        chunks = chunk_recursive(raw_text, chunk_size=512, overlap=64)
        embeddings = await embed_texts(chunks)

        chunk_rows = [
            {"id": uuid.uuid4(), "document_id": doc_id, "tid": tenant_id, "idx": i,
             "content": c, "token_count": len(c.split()), "embedding": str(e)}
            for i, (c, e) in enumerate(zip(chunks, embeddings))
        ]
        await session.execute(
            text("""INSERT INTO chunks (id, document_id, tenant_id, chunk_index, content, token_count, embedding)
                     VALUES (:id, :document_id, :tid, :idx, :content, :token_count, CAST(:embedding AS vector))"""),
            chunk_rows,
        )
        await session.commit()

    return doc_id