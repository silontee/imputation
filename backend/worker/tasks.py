import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from worker.celery_app import celery
from app.core.config import settings
from app.db.session import SessionLocal
from app.db.repository import JobRepository
from app.schemas.jobs import JobStatus
from app.services.imputation.mice import MICEImputer
from app.services.imputation.knn import KNNMethodImputer
from app.services.imputation.statistical import MeanModeImputer
from app.services.csv_reader import read_csv_with_fallback, ENCODING_CANDIDATES
from app.services.imputation.naomi import NAOMIImputer
from app.services.imputation.base import JobCanceledError
from app.services.imputation.totem import TOTEMImputer
from app.services.augmentation.smotenc import run_smotenc_augmentation


def get_imputer(model_type: str, params: dict):
    """Get the appropriate imputer based on model type."""
    imputers = {
        "MICE": MICEImputer,
        "KNN": KNNMethodImputer,
        "MEAN": MeanModeImputer,
        "REGRESSION": MICEImputer,
        "NAOMI": NAOMIImputer,
        "TOTEM": TOTEMImputer,
    }
    imputer_class = imputers.get(model_type, MeanModeImputer)
    return imputer_class(params)


def generate_imputation_preview(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    columns_to_impute: List[str],
    datetime_col: Optional[str] = None,
    max_dates: Optional[int] = None,
    max_points_per_date: int = 100,
) -> Dict[str, Any]:
    """Generate preview data for visualization.
    
    Returns a dict with:
    - dates_with_missing: list of date strings that had missing values
    - preview_data: dict mapping date -> {column_name, timestamps, original, imputed}
    """
    preview = {
        "dates_with_missing": [],
        "preview_data": {},
    }
    
    if not columns_to_impute:
        return preview

    # Pick a target column that actually has missing values
    first_col = None
    for col in columns_to_impute:
        if col in original_df.columns and original_df[col].isna().any():
            first_col = col
            break
    if first_col is None:
        # Fallback: use the first provided column if it exists
        first_col = next((c for c in columns_to_impute if c in original_df.columns), None)
    if first_col is None:
        return preview
    
    # Find datetime column if not specified
    if datetime_col is None:
        for col in original_df.columns:
            if original_df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(original_df[col].dropna().head(10))
                    datetime_col = col
                    break
                except:
                    continue
    
    # If still no datetime column, use index
    if datetime_col is None:
        # Create a simple index-based preview
        missing_mask = original_df[first_col].isna()
        
        if missing_mask.any():
            # Get first 100 rows with context around missing values
            missing_indices = missing_mask[missing_mask].index.tolist()
            if missing_indices:
                start_idx = max(0, missing_indices[0] - 10)
                end_idx = min(len(original_df), start_idx + max_points_per_date)
                
                preview["dates_with_missing"] = ["index"]
                preview["preview_data"]["index"] = {
                    "column_name": first_col,
                    "timestamps": [str(i) for i in range(start_idx, end_idx)],
                    "original": [
                        None if pd.isna(v) else float(v) 
                        for v in original_df[first_col].iloc[start_idx:end_idx].tolist()
                    ],
                    "imputed": [
                        float(v) if not pd.isna(v) else None
                        for v in result_df[first_col].iloc[start_idx:end_idx].tolist()
                    ],
                }
        return preview
    
    # Parse datetime column
    try:
        original_df = original_df.copy()
        original_df['_dt_parsed'] = pd.to_datetime(original_df[datetime_col])
        original_df['_date_str'] = original_df['_dt_parsed'].dt.strftime('%Y-%m-%d')
        result_df = result_df.copy()
        result_df['_dt_parsed'] = pd.to_datetime(result_df[datetime_col])
    except:
        return preview
    
    # Find dates with missing values
    missing_mask = original_df[first_col].isna()
    dates_with_missing = original_df.loc[missing_mask, '_date_str'].unique().tolist()
    
    if not dates_with_missing:
        return preview
    
    # Limit to max_dates if provided
    if max_dates is not None:
        dates_with_missing = dates_with_missing[:max_dates]
    preview["dates_with_missing"] = dates_with_missing
    
    # Generate preview for each date
    for date_str in dates_with_missing:
        date_mask = original_df['_date_str'] == date_str
        date_original = original_df[date_mask].head(max_points_per_date)
        date_result = result_df.loc[date_original.index]
        
        timestamps = date_original['_dt_parsed'].dt.strftime('%Y-%m-%d %H:%M').tolist()
        original_values = [
            None if pd.isna(v) else float(v) 
            for v in date_original[first_col].tolist()
        ]
        imputed_values = [
            float(v) if not pd.isna(v) else None
            for v in date_result[first_col].tolist()
        ]
        
        preview["preview_data"][date_str] = {
            "column_name": first_col,
            "timestamps": timestamps,
            "original": original_values,
            "imputed": imputed_values,
        }
    
    return preview


@celery.task(name="worker.tasks.run_imputation", bind=True, max_retries=2)
def run_imputation(self, job_id: str):
    """Run the imputation job as a Celery task."""
    db = SessionLocal()
    try:
        repo = JobRepository(db)
        job = repo.get(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}
        if job.status == JobStatus.CANCELED.value:
            return {"status": "canceled", "job_id": job_id}

        repo.update(job_id, status=JobStatus.PROCESSING.value, progress=5, stage="Reading data")
        repo.add_log(job_id, "Starting imputation process")

        # Read the input file
        try:
            df, enc = read_csv_with_fallback(job.input_path)
            repo.add_log(job_id, f"CSV encoding detected: {enc}")
            repo.add_log(job_id, "Missing values normalized: 'NA'")
        except Exception as e:
            if isinstance(e, (UnicodeDecodeError, ValueError)):
                tried = ", ".join(ENCODING_CANDIDATES)
                raise Exception(
                    f"Failed to parse CSV with supported encodings ({tried}): {str(e)}"
                )
            raise
        original_df = df.copy()  # Keep original for preview generation
        repo.update(job_id, progress=15, stage="Preprocessing")
        repo.add_log(job_id, f"Loaded {len(df)} rows")

        # Get columns by role (backward compat: fall back to action field)
        column_config = job.column_config or []
        target_columns = []
        feature_columns = []
        column_types = {}
        for col in column_config:
            role = col.get("role")
            if role is None:
                # Legacy: derive from action
                role = "TARGET" if col.get("action") == "IMPUTE" else "IGNORE"
            column_types[col["name"]] = col["type"]
            if role == "TARGET":
                target_columns.append(col["name"])
            elif role == "FEATURE":
                feature_columns.append(col["name"])

        columns_to_impute = target_columns

        repo.add_log(job_id, f"Target columns: {len(target_columns)}, Feature columns: {len(feature_columns)}")
        repo.update(job_id, progress=25, stage="Encoding")

        # Get imputer (with NAOMI progress callback if applicable)
        model_type = job.model_type
        model_params = job.model_params or {}

        def naomi_progress(epoch, total_epochs, loss_val, _model=None):
            pct = 35 + int((epoch / total_epochs) * 20)
            if epoch % max(1, total_epochs // 10) == 0 or epoch == total_epochs:
                repo.update(job_id, progress=pct, stage="Imputing")
                repo.add_log(job_id, f"Epoch {epoch}/{total_epochs} - loss: {loss_val:.6f}")

        def naomi_impute_progress(step, total_steps, preview_df):
            pct = 55 + int((step / total_steps) * 15)
            imputation_preview = generate_imputation_preview(
                original_df, preview_df, columns_to_impute
            )
            repo.update(
                job_id,
                progress=pct,
                stage="Imputing",
                imputation_preview=imputation_preview,
            )

        def totem_progress(step, total_steps, loss_val, _model=None):
            pct = 35 + int((step / total_steps) * 35)
            if step % max(1, total_steps // 10) == 0 or step == total_steps:
                repo.update(job_id, progress=pct, stage="Imputing")
                repo.add_log(job_id, f"Column {step}/{total_steps} completed")

        def totem_impute_progress(step, total_steps, preview_df):
            pct = 35 + int((step / total_steps) * 35)
            imputation_preview = generate_imputation_preview(
                original_df, preview_df, columns_to_impute
            )
            repo.update(
                job_id,
                progress=pct,
                stage="Imputing",
                imputation_preview=imputation_preview,
            )

        def _is_canceled() -> bool:
            latest = repo.get(job_id)
            return latest is not None and latest.status == JobStatus.CANCELED.value

        if model_type == "NAOMI":
            imputer = NAOMIImputer(
                params=model_params,
                progress_callback=naomi_progress,
                impute_callback=naomi_impute_progress,
                log_callback=lambda msg: repo.add_log(job_id, msg),
                cancel_callback=_is_canceled,
            )
        elif model_type == "TOTEM":
            imputer = TOTEMImputer(
                params=model_params,
                progress_callback=totem_progress,
                impute_callback=totem_impute_progress,
                log_callback=lambda msg: repo.add_log(job_id, msg),
            )
        else:
            imputer = get_imputer(model_type, model_params)

        repo.add_log(job_id, f"Using {imputer.name} imputer")
        repo.update(job_id, progress=35, stage="Imputing")

        # Run imputation (NAOMI/TOTEM can use feature_columns)
        if model_type in ("NAOMI", "TOTEM"):
            feat = feature_columns if feature_columns else None
            result_df = imputer.fit_transform(
                df, columns_to_impute, column_types, feature_columns=feat
            )
        else:
            result_df = imputer.fit_transform(df, columns_to_impute, column_types)
        repo.update(job_id, progress=70, stage="Generating preview")
        repo.add_log(job_id, "Imputation completed")

        # Generate preview data for visualization
        imputation_preview = generate_imputation_preview(
            original_df, result_df, columns_to_impute
        )
        repo.add_log(job_id, f"Preview generated for {len(imputation_preview.get('dates_with_missing', []))} dates")
        repo.update(job_id, progress=80, stage="Exporting")

        # Save result
        output_path = settings.RESULT_DIR / f"{job_id}_output.csv"
        result_df.to_csv(output_path, index=False)
        repo.add_log(job_id, f"Result saved to {output_path.name}")

        # Update job status with preview
        repo.update(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=100,
            stage="Complete",
            output_path=str(output_path),
            imputation_preview=imputation_preview,
        )
        repo.add_log(job_id, "Job completed successfully")

        return {"status": "completed", "job_id": job_id}

    except JobCanceledError:
        repo2 = JobRepository(SessionLocal())
        try:
            repo2.update(
                job_id,
                status=JobStatus.CANCELED.value,
                stage="Canceled",
            )
            repo2.add_log(job_id, "Job canceled during NAOMI processing")
        finally:
            repo2.db.close()
        return {"status": "canceled", "job_id": job_id}
    except Exception as e:
        db2 = SessionLocal()
        try:
            repo2 = JobRepository(db2)
            current = repo2.get(job_id)
            if current and current.status == JobStatus.CANCELED.value:
                repo2.update(job_id, status=JobStatus.CANCELED.value, stage="Canceled")
                repo2.add_log(job_id, "Job canceled by user")
                return {"status": "canceled", "job_id": job_id}
            repo2.update(
                job_id,
                status=JobStatus.FAILED.value,
                error_message=str(e),
                stage="Failed",
            )
            repo2.add_log(job_id, f"Error: {str(e)}")
        finally:
            db2.close()
        return {"status": "failed", "job_id": job_id, "error": str(e)}

    finally:
        db.close()


@celery.task(name="worker.tasks.run_augmentation", bind=True, max_retries=2)
def run_augmentation(self, job_id: str):
    """Run SMOTENC augmentation as a Celery task."""
    db = SessionLocal()
    try:
        repo = JobRepository(db)
        job = repo.get(job_id)
        if not job:
            return {"error": f"Job {job_id} not found"}

        repo.update(
            job_id,
            augment_status="PROCESSING",
            augment_progress=5,
            augment_stage="Loading data",
        )
        repo.add_log(job_id, "[Augment] Starting SMOTENC augmentation")

        # Load the imputed result CSV
        output_path = job.output_path
        if not output_path:
            raise Exception("No imputation result found. Run imputation first.")

        df, enc = read_csv_with_fallback(output_path)
        repo.add_log(job_id, f"[Augment] Loaded imputed data: {len(df)} rows, encoding={enc}")
        repo.update(job_id, augment_progress=10, augment_stage="Preparing")

        # Get augment params
        params = job.augment_params or {}
        label_column = params.get("label_column")
        feature_columns = params.get("feature_columns", [])
        categorical_feature_columns = params.get("categorical_feature_columns", [])
        window_size = params.get("window_size", 48)
        stride = params.get("stride", 4)
        k_neighbors = params.get("k_neighbors", 5)
        sampling_strategy = params.get("sampling_strategy", "auto")
        random_state = params.get("random_state", 42)

        if not label_column:
            raise Exception("label_column is required for augmentation")
        if not feature_columns:
            raise Exception("feature_columns is required for augmentation")

        def aug_progress(step, total, msg):
            pct = 10 + int((step / total) * 80)
            repo.update(
                job_id,
                augment_progress=pct,
                augment_stage=msg,
            )

        def aug_log(msg):
            repo.add_log(job_id, f"[Augment] {msg}")

        # Run SMOTENC
        result = run_smotenc_augmentation(
            df=df,
            label_column=label_column,
            feature_columns=feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            window_size=window_size,
            stride=stride,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            progress_callback=aug_progress,
            log_callback=aug_log,
        )

        augmented_df = result["augmented_df"]
        preview = result["preview"]
        preview["class_distribution"]["after"] = result["class_distribution_after"]

        # Save augmented CSV
        aug_output_path = settings.RESULT_DIR / f"{job_id}_smotenc.csv"
        augmented_df.to_csv(aug_output_path, index=False)
        repo.add_log(job_id, f"[Augment] Result saved to {aug_output_path.name}")

        # Update job with results
        repo.update(
            job_id,
            augment_status="COMPLETED",
            augment_progress=100,
            augment_stage="Complete",
            augment_output_path=str(aug_output_path),
            augment_preview=preview,
        )
        repo.add_log(job_id, "[Augment] Augmentation completed successfully")

        return {"status": "completed", "job_id": job_id}

    except Exception as e:
        db2 = SessionLocal()
        try:
            repo2 = JobRepository(db2)
            repo2.update(
                job_id,
                augment_status="FAILED",
                augment_error=str(e),
                augment_stage="Failed",
            )
            repo2.add_log(job_id, f"[Augment] Error: {str(e)}")
        finally:
            db2.close()
        return {"status": "failed", "job_id": job_id, "error": str(e)}

    finally:
        db.close()
