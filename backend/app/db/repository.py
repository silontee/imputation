"""Database-backed job repository, replacing the in-memory JobStore."""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import ImputationJob
from app.schemas.jobs import JobStatus


class JobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, filename: str, input_path: str, file_size_bytes: int = 0) -> ImputationJob:
        job_id = uuid.uuid4()
        job = ImputationJob(
            id=job_id,
            status=JobStatus.UPLOADED.value,
            original_filename=filename,
            file_size_bytes=file_size_bytes,
            input_path=input_path,
            logs=[],
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get(self, job_id: str) -> Optional[ImputationJob]:
        uid = uuid.UUID(job_id) if isinstance(job_id, str) else job_id
        return self.db.query(ImputationJob).filter(ImputationJob.id == uid).first()

    def update(self, job_id: str, **kwargs) -> Optional[ImputationJob]:
        job = self.get(job_id)
        if not job:
            return None

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        job.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(job)
        return job

    def add_log(self, job_id: str, message: str) -> None:
        job = self.get(job_id)
        if not job:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        logs = list(job.logs) if job.logs else []
        logs.append(f"[{timestamp}] {message}")
        if len(logs) > 50:
            logs = logs[-50:]

        job.logs = logs
        job.updated_at = datetime.utcnow()
        self.db.commit()

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        self.db.delete(job)
        self.db.commit()
        return True

    def list_all(self) -> List[ImputationJob]:
        return self.db.query(ImputationJob).order_by(ImputationJob.created_at.desc()).all()
