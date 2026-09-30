from celery import Celery
from app.config import settings

celery_app = Celery(
    "rag_pipeline",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,           
    worker_prefetch_multiplier=1,  
    beat_schedule={
        "resync-sources-every-6-hours": {
            "task": "app.tasks.resync_all_sources_task",
            "schedule": 6 * 60 * 60,
        },
    },
)