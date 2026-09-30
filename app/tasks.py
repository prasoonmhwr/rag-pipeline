import asyncio

from fastapi import logger
from app.celery_app import celery_app
from app.services.ingestion import ingest_document

@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def ingest_document_task(self, tenant_id, source_uri, title, raw_text, metadata=None):
    try:
        return str(asyncio.run(ingest_document(tenant_id, source_uri, title, raw_text, metadata)))
    except Exception as exc:
        raise self.retry(exc=exc)

@celery_app.task
def resync_all_sources_task():
    logger.info("Scheduled resync fired — no source connector configured yet.")