import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ingestion import ingest_document

router = APIRouter()

class IngestRequest(BaseModel):
    tenant_id: str
    source_uri: str
    title: str
    raw_text: str
    metadata: dict | None = None


@router.post("/")
async def ingest_endpoint(req: IngestRequest):
    doc_id = await ingest_document(
        uuid.UUID(req.tenant_id), req.source_uri, req.title, req.raw_text, req.metadata
    )
    return {"document_id": str(doc_id), "status": "ingested"}