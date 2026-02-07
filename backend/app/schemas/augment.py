from typing import List, Optional, Union
from pydantic import BaseModel, Field


class AugmentRequest(BaseModel):
    label_column: str = Field(..., description="Label column name (categorical/class)")
    feature_columns: List[str] = Field(..., description="Feature column names to include")
    categorical_feature_columns: List[str] = Field(
        default_factory=list,
        description="Categorical feature column names (required for SMOTENC)",
    )
    window_size: int = Field(default=48, ge=4, le=1024, description="Sliding window size")
    stride: int = Field(default=4, ge=1, le=512, description="Sliding window stride")
    k_neighbors: int = Field(default=5, ge=1, le=50, description="Number of nearest neighbors")
    sampling_strategy: Union[str, float] = Field(
        default="auto",
        description="Sampling strategy: 'auto', 'minority', 'not majority', or float ratio",
    )
    random_state: int = Field(default=42, ge=0, description="Random state for reproducibility")


class AugmentResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None


class AugmentStatusResponse(BaseModel):
    job_id: str
    augment_status: str  # IDLE, QUEUED, PROCESSING, COMPLETED, FAILED
    augment_progress: int = 0
    augment_stage: str = ""
    augment_download_url: Optional[str] = None
    augment_error: Optional[str] = None
    augment_preview: Optional[dict] = None
    logs: List[str] = []
