import logging
from pydantic import BaseModel
from google import genai
from google.genai import types
from app.config import settings
from app.services.embeddings import embed_texts
from app.services.retrieval import hybrid_search


logger = logging.getLogger("rag.generation")
client = genai.Client(api_key=settings.gemini_api_key)
GENERATION_MODEL = "gemini-3.8-flash"
MIN_SIMILARITY_THRESHOLD = 0.5  

class Citation(BaseModel):
    chunk_id: str
    quote: str

class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    sufficient_context: bool

SYSTEM_PROMPT = """You answer ONLY using the provided context chunks. Rules:
1. Every factual sentence must be supported by at least one chunk.
2. Cite the chunk_id for each claim.
3. For each citation's "quote" field, copy a short EXACT VERBATIM excerpt —
   a phrase or sentence copied character-for-character from that chunk's text.
   Do not paraphrase, summarize, or reword it in any way.
4. If context is insufficient, set sufficient_context=false and say so — do not
   fill gaps from outside knowledge.
5. Never state something the context contradicts."""

async def generate_grounded_answer(query: str, chunks: list[dict]) -> GroundedAnswer:
    context_block = "\n\n".join(f"[chunk_id={c['id']}]\n{c['content']}" for c in chunks)
    resp = await client.aio.models.generate_content(
        model=GENERATION_MODEL,
        contents=f"Context:\n{context_block}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=GroundedAnswer,
        ),
    )
    return resp.parsed

def verify_citations(answer: GroundedAnswer, retrieved_chunks: dict[str, str]) -> list[str]:
    problems = []
    valid_ids = set(retrieved_chunks.keys())
    for c in answer.citations:
        if c.chunk_id not in valid_ids:
            problems.append(f"Cited chunk_id {c.chunk_id} was never retrieved (hallucinated citation).")
            continue
        normalized_quote = " ".join(c.quote.strip().split())
        normalized_chunk = " ".join(retrieved_chunks[c.chunk_id].split())
        if normalized_quote and normalized_quote not in normalized_chunk:
            problems.append(f"Quote for {c.chunk_id} doesn't actually appear in that chunk.")
    return problems


async def answer_with_abstention(query: str, session, tenant_id: str) -> dict:
    [query_embedding] = await embed_texts([query])
    results = await hybrid_search(session, tenant_id, query, query_embedding, top_k=8)
    top_similarity = results[0]["vector_similarity"] if results else 0.0
    logger.info("query=%r top_vector_similarity=%.4f", query, top_similarity)

    if not results or top_similarity < MIN_SIMILARITY_THRESHOLD:
        return {"answer": "I don't have enough information to answer that.", "citations": [], "abstained": True}

    chunk_map = {str(r["id"]): r["content"] for r in results}
    answer = await generate_grounded_answer(query, results)
    problems = verify_citations(answer, chunk_map)
    logger.info("sufficient_context=%s problems=%s raw_answer=%r citations=%s",
                answer.sufficient_context, problems, answer.answer, answer.citations)

    if not answer.sufficient_context or problems:
        return {"answer": "I found related info but can't confidently answer.", "citations": [],
                "abstained": True, "debug_problems": problems if problems else ["model set sufficient_context=false"]}

    return {"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations], "abstained": False}