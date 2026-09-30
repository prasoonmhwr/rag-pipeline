from fastapi import APIRouter, Depends
from pydantic import BaseModel
from app.dependencies import get_db_session_for_user
from app.services.generation import answer_with_abstention

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@router.post("/")
async def query_endpoint(req: QueryRequest, session=Depends(get_db_session_for_user)):
    return await answer_with_abstention(
        req.query, session, "00000000-0000-0000-0000-000000000001"
    )