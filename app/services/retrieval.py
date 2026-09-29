from sqlalchemy import text

async def hybrid_search(session, tenant_id, query_text, query_embedding, top_k=10, rrf_k=60):
    result = await session.execute(
        text("""
            WITH vector_results AS (
                SELECT id, RANK() OVER (ORDER BY embedding <=> CAST(:query_embedding AS vector)) AS rank
                FROM chunks WHERE tenant_id = :tenant_id
                ORDER BY embedding <=> CAST(:query_embedding AS vector) LIMIT 50
            ),
            text_results AS (
                SELECT id, RANK() OVER (ORDER BY ts_rank_cd(content_tsv, query) DESC) AS rank
                FROM chunks, plainto_tsquery('english', :query_text) query
                WHERE tenant_id = :tenant_id AND content_tsv @@ query
                ORDER BY ts_rank_cd(content_tsv, query) DESC LIMIT 50
            )
            SELECT c.id, c.content, c.document_id, d.title,
                COALESCE(1.0 / (:rrf_k + v.rank), 0.0) + COALESCE(1.0 / (:rrf_k + t.rank), 0.0) AS rrf_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN vector_results v ON v.id = c.id
            LEFT JOIN text_results t ON t.id = c.id
            WHERE v.id IS NOT NULL OR t.id IS NOT NULL
            ORDER BY rrf_score DESC LIMIT :top_k
        """),
        {"query_embedding": str(query_embedding), "tenant_id": tenant_id,
         "query_text": query_text, "rrf_k": rrf_k, "top_k": top_k},
    )
    return [dict(row._mapping) for row in result]