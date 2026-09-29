import asyncio
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

client = genai.Client(api_key=settings.gemini_api_key)
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 1536          # must match the vector(1536) column in schema.sql
MAX_BATCH_SIZE = 100

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    resp = await client.aio.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [item.values for item in resp.embeddings]

async def embed_texts(texts: list[str], concurrency: int = 5) -> list[list[float]]:
    batches = [texts[i:i + MAX_BATCH_SIZE] for i in range(0, len(texts), MAX_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(batch):
        async with semaphore:
            return await _embed_batch(batch)

    results = await asyncio.gather(*[_run(b) for b in batches])
    return [emb for batch_result in results for emb in batch_result]