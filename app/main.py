from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from app.db import engine
from app.routers import ingest, query

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    yield
    await engine.dispose()

app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])