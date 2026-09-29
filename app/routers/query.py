from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.embeddings import embed_texts
from app.services.retrieval import hybrid_search
from app.db import AsyncSessionLocal

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@router.post("/")
async def query_endpoint(req: QueryRequest):
    async with AsyncSessionLocal() as session:
        [query_embedding] = await embed_texts([req.query])
        results = await hybrid_search(session, "00000000-0000-0000-0000-000000000001",
                                       req.query, query_embedding, top_k=req.top_k)
    return results