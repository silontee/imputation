from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.jobs import (
    StartJobRequest,
    StartJobResponse,
    JobStatusResponse,
    JobStatus,
)
from app.db.session import get_db
from app.db.repository import JobRepository
from app.worker_client import enqueue_imputation_task

router = APIRouter(tags=["jobs"])


@router.post("/jobs/{job_id}/start", response_model=StartJobResponse, status_code=202)
async def start_job(
    job_id: str,
    request: StartJobRequest,
    db: Session = Depends(get_db),
):
    """Start an imputation job."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [JobStatus.UPLOADED.value, JobStatus.REVIEWED.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start job with status: {job.status}"
        )

    # Update job configuration
    repo.update(
        job_id,
        status=JobStatus.QUEUED.value,
        model_type=request.model_type.value,
        model_params=request.hyperparameters,
        column_config=[col.model_dump() for col in request.column_config],
        progress=0,
        stage="Queued",
    )
    repo.add_log(job_id, f"Job queued with model: {request.model_type.value}")

    # Enqueue Celery task
    enqueue_imputation_task(job_id)

    return StartJobResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get the status of a job."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    download_url = None
    if job.status == JobStatus.COMPLETED.value and job.output_path:
        download_url = f"/api/v1/jobs/{job_id}/download"

    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        progress=job.progress,
        stage=job.stage or "",
        download_url=download_url,
        error_message=job.error_message,
        logs=(job.logs or [])[-20:],
        imputation_preview=job.imputation_preview,
    )


@router.get("/jobs/{job_id}/download")
async def download_result(job_id: str, db: Session = Depends(get_db)):
    """Download the imputed result file."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed yet")

    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=job.output_path,
        filename=f"imputed_{job.original_filename}",
        media_type="text/csv",
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """Cancel a job."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
        raise HTTPException(status_code=400, detail="Cannot cancel finished job")

    repo.update(job_id, status=JobStatus.CANCELED.value, stage="Canceled")
    repo.add_log(job_id, "Job canceled by user")

    return {"message": "Job canceled"}
