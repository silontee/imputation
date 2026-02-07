"""
SMOTENC-based time series augmentation service.

Flow:
  1) Load imputed DataFrame
  2) Sliding window → feature vectors + label
  3) Identify categorical feature indices for SMOTENC
  4) Apply SMOTENC to generate synthetic windows
  5) Reconstruct synthetic windows back to time series format
  6) Return augmented DataFrame + preview data
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

logger = logging.getLogger(__name__)


def create_sliding_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    window_size: int = 48,
    stride: int = 4,
) -> tuple:
    """Convert time series DataFrame to sliding window feature vectors.

    Returns:
        X: np.ndarray of shape (n_windows, window_size * n_features)
        y: np.ndarray of shape (n_windows,)
        feature_names: list of flattened feature names
    """
    features = df[feature_columns].values  # (T, n_features)
    labels = df[label_column].values  # (T,)

    n_samples = len(df)
    n_features = len(feature_columns)

    windows_X = []
    windows_y = []

    for start in range(0, n_samples - window_size + 1, stride):
        end = start + window_size
        window = features[start:end]  # (window_size, n_features)
        windows_X.append(window.flatten())  # (window_size * n_features,)
        # Use the label of the last timestep in the window
        windows_y.append(labels[end - 1])

    X = np.array(windows_X)
    y = np.array(windows_y)

    # Generate feature names for the flattened vector
    feature_names = []
    for t in range(window_size):
        for col in feature_columns:
            feature_names.append(f"{col}_t{t}")

    return X, y, feature_names


def get_categorical_indices(
    feature_columns: List[str],
    categorical_feature_columns: List[str],
    window_size: int,
) -> List[int]:
    """Calculate categorical feature indices in the flattened window vector.

    Each window has window_size timesteps * n_features columns.
    If a feature is categorical, all its timestep copies are categorical.
    """
    n_features = len(feature_columns)
    cat_indices = []

    for t in range(window_size):
        for feat_idx, col in enumerate(feature_columns):
            if col in categorical_feature_columns:
                flat_idx = t * n_features + feat_idx
                cat_indices.append(flat_idx)

    return cat_indices


def run_smotenc_augmentation(
    df: pd.DataFrame,
    label_column: str,
    feature_columns: List[str],
    categorical_feature_columns: List[str],
    window_size: int = 48,
    stride: int = 4,
    k_neighbors: int = 5,
    sampling_strategy: Union[str, float] = "auto",
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run SMOTENC augmentation on time series data.

    Args:
        df: Input DataFrame (imputed, no missing values expected)
        label_column: Name of the label/class column
        feature_columns: List of feature column names
        categorical_feature_columns: List of categorical feature column names
        window_size: Sliding window size
        stride: Sliding window stride
        k_neighbors: Number of nearest neighbors for SMOTENC
        sampling_strategy: SMOTENC sampling strategy
        random_state: Random seed
        progress_callback: Optional callback(step, total, message)
        log_callback: Optional callback(message)

    Returns:
        dict with:
          - augmented_df: pd.DataFrame with original + synthetic rows
          - original_count: number of original windows
          - synthetic_count: number of synthetic windows
          - class_distribution_before: dict
          - class_distribution_after: dict
          - preview: dict for visualization
    """

    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    def _progress(step: int, total: int, msg: str):
        if progress_callback:
            progress_callback(step, total, msg)

    total_steps = 5
    _progress(0, total_steps, "Preparing data")
    _log(f"Starting SMOTENC augmentation: {len(df)} rows, window={window_size}, stride={stride}")

    # --- Step 1: Validate ---
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame")
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns not found: {missing_cols}")

    # Separate numeric-only feature columns for SMOTENC
    numeric_features = [c for c in feature_columns if c not in categorical_feature_columns]

    _progress(1, total_steps, "Creating sliding windows")
    _log(f"Features: {len(feature_columns)} ({len(numeric_features)} numeric, {len(categorical_feature_columns)} categorical)")

    # --- Step 2: Create sliding windows ---
    X, y, feat_names = create_sliding_windows(
        df, feature_columns, label_column, window_size, stride
    )
    _log(f"Created {len(X)} windows from {len(df)} rows")

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist_before = dict(zip([str(u) for u in unique], [int(c) for c in counts]))
    _log(f"Class distribution (before): {class_dist_before}")

    if len(unique) < 2:
        raise ValueError(
            f"Need at least 2 classes for SMOTENC, but found {len(unique)}: {unique.tolist()}"
        )

    # Check minimum samples per class
    min_count = int(counts.min())
    if min_count < k_neighbors + 1:
        _log(f"Warning: min class count ({min_count}) < k_neighbors+1 ({k_neighbors + 1}), adjusting k_neighbors to {max(1, min_count - 1)}")
        k_neighbors = max(1, min_count - 1)

    # --- Step 3: Compute categorical indices ---
    _progress(2, total_steps, "Computing categorical indices")

    if categorical_feature_columns:
        cat_indices = get_categorical_indices(
            feature_columns, categorical_feature_columns, window_size
        )
        _log(f"Categorical indices: {len(cat_indices)} positions in flattened vector")
    else:
        # If no categorical columns, use SMOTENC with a dummy categorical
        # or fall back. SMOTENC requires at least one categorical feature.
        # We'll add a dummy categorical column.
        _log("No categorical features specified. Adding label as categorical feature for SMOTENC.")
        # Append label as the last feature in each window
        X = np.column_stack([X, y])
        cat_indices = [X.shape[1] - 1]

    # --- Step 4: Apply SMOTENC ---
    _progress(3, total_steps, "Running SMOTENC")
    _log(f"Applying SMOTENC with k_neighbors={k_neighbors}, sampling_strategy={sampling_strategy}")

    smotenc = SMOTENC(
        categorical_features=cat_indices,
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    X_resampled, y_resampled = smotenc.fit_resample(X, y)

    # Separate original and synthetic
    n_original = len(X)
    n_synthetic = len(X_resampled) - n_original
    X_synthetic = X_resampled[n_original:]
    y_synthetic = y_resampled[n_original:]

    unique_after, counts_after = np.unique(y_resampled, return_counts=True)
    class_dist_after = dict(zip([str(u) for u in unique_after], [int(c) for c in counts_after]))
    _log(f"Class distribution (after): {class_dist_after}")
    _log(f"Generated {n_synthetic} synthetic windows")

    # --- Step 5: Reconstruct synthetic windows to DataFrame ---
    _progress(4, total_steps, "Reconstructing time series")

    # Remove dummy categorical if added
    if not categorical_feature_columns:
        X_synthetic = X_synthetic[:, :-1]

    n_features = len(feature_columns)
    synthetic_rows = []

    for i, window_flat in enumerate(X_synthetic):
        window = window_flat.reshape(window_size, n_features)
        for t in range(window_size):
            row = {}
            for f_idx, col in enumerate(feature_columns):
                row[col] = window[t, f_idx]
            row[label_column] = y_synthetic[i]
            row["_synthetic"] = True
            row["_window_id"] = i
            row["_timestep"] = t
            synthetic_rows.append(row)

    synthetic_df = pd.DataFrame(synthetic_rows)

    # Build augmented DataFrame: original + synthetic
    original_with_meta = df.copy()
    original_with_meta["_synthetic"] = False
    original_with_meta["_window_id"] = -1
    original_with_meta["_timestep"] = range(len(original_with_meta))

    augmented_df = pd.concat([original_with_meta, synthetic_df], ignore_index=True)

    # --- Generate preview data ---
    preview = _generate_augment_preview(
        df, synthetic_df, feature_columns, label_column, n_original, n_synthetic
    )

    _progress(5, total_steps, "Complete")
    _log(f"Augmentation complete: {len(df)} original + {len(synthetic_df)} synthetic = {len(augmented_df)} total rows")

    return {
        "augmented_df": augmented_df,
        "original_count": n_original,
        "synthetic_count": n_synthetic,
        "class_distribution_before": class_dist_before,
        "class_distribution_after": class_dist_after,
        "preview": preview,
    }


def _generate_augment_preview(
    original_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    feature_columns: List[str],
    label_column: str,
    n_original_windows: int,
    n_synthetic_windows: int,
    max_preview_points: int = 200,
) -> Dict[str, Any]:
    """Generate preview data for the augmentation overlay graph."""
    preview: Dict[str, Any] = {
        "original_count": len(original_df),
        "synthetic_count": len(synthetic_df),
        "original_windows": n_original_windows,
        "synthetic_windows": n_synthetic_windows,
        "columns": {},
    }

    # For each numeric feature column, generate preview time series
    for col in feature_columns[:5]:  # Limit to first 5 columns for preview
        if col in original_df.columns and col in synthetic_df.columns:
            # Sample original data
            orig_values = original_df[col].head(max_preview_points).tolist()
            orig_values = [float(v) if pd.notna(v) else None for v in orig_values]

            # Sample synthetic data (first few windows)
            synth_values = synthetic_df[col].head(max_preview_points).tolist()
            synth_values = [float(v) if pd.notna(v) else None for v in synth_values]

            preview["columns"][col] = {
                "original": orig_values,
                "synthetic": synth_values,
            }

    # Class distribution summary
    preview["class_distribution"] = {
        "before": dict(original_df[label_column].value_counts().astype(int).to_dict())
        if label_column in original_df.columns
        else {},
        "after": {},
    }

    return preview
