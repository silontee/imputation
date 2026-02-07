from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.analyze import ColumnType, ColumnAction, ColumnRole


class JobStatus(str, Enum):
    UPLOADED = "UPLOADED"
    REVIEWED = "REVIEWED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class ModelType(str, Enum):
    MICE = "MICE"
    KNN = "KNN"
    MEAN = "MEAN"
    REGRESSION = "REGRESSION"
    NAOMI = "NAOMI"
    TOTEM = "TOTEM"


# --- Model-specific hyperparameter schemas ---

MICE_ESTIMATORS = {"bayesian_ridge", "random_forest", "extra_trees"}
KNN_WEIGHTS = {"uniform", "distance"}
KNN_METRICS = {"nan_euclidean", "euclidean"}


def validate_hyperparameters(model_type: ModelType, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize hyperparameters based on model type."""
    if model_type in (ModelType.MICE, ModelType.REGRESSION):
        max_iter = params.get("max_iter", 10)
        if not isinstance(max_iter, int) or not 1 <= max_iter <= 100:
            raise ValueError("max_iter must be an integer between 1 and 100")
        random_state = params.get("random_state", 42)
        if not isinstance(random_state, int):
            raise ValueError("random_state must be an integer")
        estimator = params.get("estimator", "bayesian_ridge")
        if estimator not in MICE_ESTIMATORS:
            raise ValueError(f"estimator must be one of {MICE_ESTIMATORS}")
        return {"max_iter": max_iter, "random_state": random_state, "estimator": estimator}

    elif model_type == ModelType.KNN:
        n_neighbors = params.get("n_neighbors", 5)
        if not isinstance(n_neighbors, int) or not 1 <= n_neighbors <= 50:
            raise ValueError("n_neighbors must be an integer between 1 and 50")
        weights = params.get("weights", "uniform")
        if weights not in KNN_WEIGHTS:
            raise ValueError(f"weights must be one of {KNN_WEIGHTS}")
        metric = params.get("metric", "nan_euclidean")
        if metric not in KNN_METRICS:
            raise ValueError(f"metric must be one of {KNN_METRICS}")
        return {"n_neighbors": n_neighbors, "weights": weights, "metric": metric}

    elif model_type == ModelType.MEAN:
        return {}

    elif model_type == ModelType.NAOMI:
        hidden_dim = params.get("hidden_dim", 64)
        if not isinstance(hidden_dim, int) or not 16 <= hidden_dim <= 512:
            raise ValueError("hidden_dim must be an integer between 16 and 512")
        epochs = params.get("epochs", 50)
        if not isinstance(epochs, int) or not 1 <= epochs <= 500:
            raise ValueError("epochs must be an integer between 1 and 500")
        lr = params.get("lr", 1e-3)
        if not isinstance(lr, (int, float)) or not 1e-6 <= lr <= 1.0:
            raise ValueError("lr must be a number between 1e-6 and 1.0")
        num_resolutions = params.get("num_resolutions", None)
        if num_resolutions is not None:
            if not isinstance(num_resolutions, int) or not 1 <= num_resolutions <= 16:
                raise ValueError("num_resolutions must be an integer between 1 and 16")
        highest = params.get("highest", None)
        if highest is not None:
            if not isinstance(highest, int) or highest < 1:
                raise ValueError("highest must be a positive integer")
            # require power of two to match NAOMI step sizes
            if highest & (highest - 1) != 0:
                raise ValueError("highest must be a power of two (e.g., 2, 4, 8, 16)")
            if highest > 1024:
                raise ValueError("highest must be <= 1024")
        window_size = params.get("window_size", 50)
        if not isinstance(window_size, int) or not 4 <= window_size <= 500:
            raise ValueError("window_size must be an integer between 4 and 500")
        batch_size = params.get("batch_size", 64)
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be an integer between 1 and 1024")
        n_layers = params.get("n_layers", 2)
        if not isinstance(n_layers, int) or not 1 <= n_layers <= 8:
            raise ValueError("n_layers must be an integer between 1 and 8")
        clip = params.get("clip", 10.0)
        if not isinstance(clip, (int, float)) or not 0.1 <= float(clip) <= 100.0:
            raise ValueError("clip must be a number between 0.1 and 100.0")
        preview_updates = params.get("preview_updates", 10)
        if not isinstance(preview_updates, int) or not 1 <= preview_updates <= 100:
            raise ValueError("preview_updates must be an integer between 1 and 100")
        return {
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": float(lr),
            "num_resolutions": num_resolutions,
            "highest": highest,
            "window_size": window_size,
            "batch_size": batch_size,
            "n_layers": n_layers,
            "clip": float(clip),
            "preview_updates": preview_updates,
        }

    elif model_type == ModelType.TOTEM:
        window_size = params.get("window_size", 96)
        if not isinstance(window_size, int) or not 16 <= window_size <= 512:
            raise ValueError("window_size must be an integer between 16 and 512")
        # window_size must be divisible by compression_factor (4)
        if window_size % 4 != 0:
            raise ValueError("window_size must be divisible by 4 (compression factor)")
        normalization = params.get("normalization", "zscore")
        if normalization not in ("zscore", "minmax"):
            raise ValueError("normalization must be 'zscore' or 'minmax'")
        merge_mode = params.get("merge_mode", "non_overlap")
        if merge_mode not in ("non_overlap", "overlap"):
            raise ValueError("merge_mode must be 'non_overlap' or 'overlap'")
        stride = params.get("stride", window_size if merge_mode == "non_overlap" else max(1, window_size // 2))
        if not isinstance(stride, int):
            raise ValueError("stride must be an integer")
        if merge_mode == "overlap":
            if not (1 <= stride < window_size):
                raise ValueError("for overlap mode, stride must satisfy 1 <= stride < window_size")
        else:
            # Non-overlap mode is fixed stride = window_size.
            stride = window_size
        preview_updates = params.get("preview_updates", 10)
        if not isinstance(preview_updates, int) or not 1 <= preview_updates <= 50:
            raise ValueError("preview_updates must be an integer between 1 and 50")
        return {
            "window_size": window_size,
            "normalization": normalization,
            "merge_mode": merge_mode,
            "stride": stride,
            "preview_updates": preview_updates,
        }

    return params


class ColumnConfig(BaseModel):
    name: str
    type: ColumnType
    role: Optional[ColumnRole] = None
    # Backward compat: accept legacy "action" field
    action: Optional[ColumnAction] = None

    @model_validator(mode="after")
    def resolve_role(self):
        """If role is not set, derive from legacy action field."""
        if self.role is None:
            if self.action == ColumnAction.IMPUTE:
                self.role = ColumnRole.TARGET
            elif self.action == ColumnAction.IGNORE:
                self.role = ColumnRole.IGNORE
            else:
                self.role = ColumnRole.IGNORE
        return self


class StartJobRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_type: ModelType = Field(..., examples=["MICE"])
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    column_config: List[ColumnConfig]

    @model_validator(mode="after")
    def validate_params(self):
        self.hyperparameters = validate_hyperparameters(
            self.model_type, self.hyperparameters
        )
        # Validate: at least 1 TARGET
        targets = [c for c in self.column_config if c.role == ColumnRole.TARGET]
        if not targets:
            raise ValueError("At least one TARGET column is required")
        return self


class StartJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class ImputationPreviewData(BaseModel):
    """Preview data for a single date with missing values."""
    column_name: str
    timestamps: List[str]
    original: List[Optional[float]]
    imputed: List[Optional[float]]


class ImputationPreview(BaseModel):
    """Preview of imputation results for visualization."""
    dates_with_missing: List[str] = Field(default_factory=list)
    preview_data: Dict[str, ImputationPreviewData] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: int = 0
    stage: str = ""
    download_url: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    imputation_preview: Optional[ImputationPreview] = None
