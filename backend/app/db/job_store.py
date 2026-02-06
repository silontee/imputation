from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import uuid

from app.schemas.jobs import JobStatus


@dataclass
class Job:
    id: str
    status: JobStatus
    filename: str
    input_path: str
    output_path: Optional[str] = None
    model_type: Optional[str] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    column_config: List[Dict[str, Any]] = field(default_factory=list)
    progress: int = 0
    stage: str = ""
    error_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class JobStore:
    """In-memory job storage. Can be replaced with database later."""

    def __init__(self):
        self._jobs: Dict[str, Job] = {}

    def create(self, filename: str, input_path: str) -> Job:
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            status=JobStatus.UPLOADED,
            filename=filename,
            input_path=input_path,
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        job.updated_at = datetime.now()
        return job

    def add_log(self, job_id: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            timestamp = datetime.now().strftime("%H:%M:%S")
            job.logs.append(f"[{timestamp}] {message}")
            # Keep only last 50 logs
            if len(job.logs) > 50:
                job.logs = job.logs[-50:]

    def delete(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    def list_all(self) -> List[Job]:
        return list(self._jobs.values())


# Global job store instance
job_store = JobStore()
