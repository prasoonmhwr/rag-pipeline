from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.services.embeddings import embed_texts
from app.services.retrieval import hybrid_search
from app.db import AsyncSessionLocal
from app.services.generation import answer_with_abstention

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@router.post("/")
async def query_endpoint(req: QueryRequest):
    async with AsyncSessionLocal() as session:
        return await answer_with_abstention(
            req.query, session, "00000000-0000-0000-0000-000000000001"
        )
    return results