"""Client for enqueuing Celery tasks from the FastAPI app."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def enqueue_imputation_task(job_id: str):
    """Send an imputation task to the Celery worker."""
    celery_app.send_task("worker.tasks.run_imputation", args=[job_id])
