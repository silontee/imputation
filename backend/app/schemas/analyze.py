from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class ColumnType(str, Enum):
    ID = "ID"
    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"
    DATETIME = "DATETIME"


class ColumnAction(str, Enum):
    IMPUTE = "IMPUTE"
    IGNORE = "IGNORE"


class ColumnRole(str, Enum):
    TARGET = "TARGET"
    FEATURE = "FEATURE"
    IGNORE = "IGNORE"


class ColumnProfile(BaseModel):
    name: str
    detected_type: ColumnType
    null_count: int
    null_ratio: float
    unique_count: int
    example: List[Any] = Field(default_factory=list)
    recommended_action: ColumnAction
    warnings: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    job_id: str
    filename: str
    sample_rows: int
    total_null_ratio: float
    columns: List[ColumnProfile]
