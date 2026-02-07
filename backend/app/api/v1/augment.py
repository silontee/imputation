from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.schemas.augment import AugmentRequest, AugmentResponse, AugmentStatusResponse
from app.db.session import get_db
from app.db.repository import JobRepository
from app.worker_client import enqueue_augmentation_task

router = APIRouter(tags=["augmentation"])


@router.post("/jobs/{job_id}/augment", response_model=AugmentResponse, status_code=202)
async def start_augmentation(
    job_id: str,
    request: AugmentRequest,
    db: Session = Depends(get_db),
):
    """Start SMOTENC augmentation on a completed imputation job."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "COMPLETED":
        raise HTTPException(
            status_code=400,
            detail=f"Imputation must be completed first. Current status: {job.status}",
        )

    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=400, detail="Imputation result file not found")

    # Validate label column exists in the data
    # (lightweight check - full validation happens in the worker)
    if not request.label_column:
        raise HTTPException(status_code=400, detail="label_column is required")
    if not request.feature_columns:
        raise HTTPException(status_code=400, detail="feature_columns is required")

    # Store augment params and enqueue
    augment_params = request.model_dump()
    # Convert sampling_strategy to proper type for JSON serialization
    if isinstance(augment_params.get("sampling_strategy"), float):
        augment_params["sampling_strategy"] = float(augment_params["sampling_strategy"])

    repo.update(
        job_id,
        augment_status="QUEUED",
        augment_params=augment_params,
        augment_progress=0,
        augment_stage="Queued",
        augment_error=None,
    )
    repo.add_log(job_id, f"[Augment] Augmentation queued: label={request.label_column}, window={request.window_size}")

    enqueue_augmentation_task(job_id)

    return AugmentResponse(
        job_id=job_id,
        status="QUEUED",
        message="Augmentation task queued",
    )


@router.get("/jobs/{job_id}/augment", response_model=AugmentStatusResponse)
async def get_augment_status(job_id: str, db: Session = Depends(get_db)):
    """Get augmentation status for a job."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    download_url = None
    if job.augment_status == "COMPLETED" and job.augment_output_path:
        download_url = f"/api/v1/jobs/{job_id}/augment/download"

    return AugmentStatusResponse(
        job_id=str(job.id),
        augment_status=job.augment_status or "IDLE",
        augment_progress=job.augment_progress or 0,
        augment_stage=job.augment_stage or "",
        augment_download_url=download_url,
        augment_error=job.augment_error,
        augment_preview=job.augment_preview,
        logs=(job.logs or [])[-20:],
    )


@router.get("/jobs/{job_id}/augment/download")
async def download_augmented_result(job_id: str, db: Session = Depends(get_db)):
    """Download the augmented result CSV."""
    repo = JobRepository(db)
    job = repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.augment_status != "COMPLETED":
        raise HTTPException(status_code=400, detail="Augmentation not completed yet")

    if not job.augment_output_path or not Path(job.augment_output_path).exists():
        raise HTTPException(status_code=404, detail="Augmented result file not found")

    return FileResponse(
        path=job.augment_output_path,
        filename=f"augmented_{job.original_filename}",
        media_type="text/csv",
    )
