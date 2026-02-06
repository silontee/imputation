"""TOTEM VQVAE-based time series imputer."""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.services.imputation.base import BaseImputer
from app.services.imputation.totem.model_loader import TOTEMModelLoader
from app.services.imputation.totem.inference import (
    revintime2codes,
    codes2timerevin,
    apply_zscore_normalization,
    apply_minmax_normalization,
    inverse_zscore,
    inverse_minmax,
)

logger = logging.getLogger(__name__)


class TOTEMImputer(BaseImputer):
    """TOTEM VQVAE-based univariate time series imputer.

    Uses a pretrained VQVAE tokenizer to encode time series into discrete codes,
    then decodes back to reconstruct missing values.

    Params (via hyperparameters dict):
        window_size: int (default 96) - must be divisible by 4
        normalization: str (default "zscore") - "zscore" or "minmax"
        merge_mode: str (default "non_overlap") - "non_overlap" or "overlap"
        stride: int (default window_size for non_overlap, window_size//2 for overlap)
        preview_updates: int (default 10) - number of intermediate previews
    """

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        progress_callback: Optional[Callable[[int, int, float, Any], None]] = None,
        impute_callback: Optional[Callable[[int, int, pd.DataFrame], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(params)
        self.progress_callback = progress_callback
        self.impute_callback = impute_callback
        self.log_callback = log_callback

    @property
    def name(self) -> str:
        return "TOTEM"

    def _log(self, msg: str) -> None:
        """Log a message via callback if available."""
        logger.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns_to_impute: List[str],
        column_types: Dict[str, str],
        feature_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        """Impute missing values using TOTEM VQVAE.

        Each target column is processed independently (univariate).

        Args:
            df: Input dataframe
            columns_to_impute: TARGET columns to impute
            column_types: Column type mapping
            feature_columns: Not used (TOTEM is univariate)

        Returns:
            DataFrame with imputed values
        """
        result = df.copy()

        if not columns_to_impute:
            return result

        # Filter to numeric columns that exist
        valid_columns = []
        for col in columns_to_impute:
            if col not in df.columns:
                self._log(f"Column {col} not found, skipping")
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                self._log(f"Column {col} is not numeric, skipping")
                continue
            if not df[col].isna().any():
                self._log(f"Column {col} has no missing values, skipping")
                continue
            valid_columns.append(col)

        if not valid_columns:
            self._log("No valid columns to impute")
            return result

        # Load model
        model_path = settings.TOTEM_MODEL_PATH
        config_path = settings.TOTEM_CONFIG_PATH if settings.TOTEM_CONFIG_PATH else None
        code_path = settings.TOTEM_CODE_PATH if settings.TOTEM_CODE_PATH else None
        self._log(f"Loading TOTEM model from {model_path}")

        try:
            model, config = TOTEMModelLoader.load(
                model_path=model_path,
                config_path=config_path,
                code_path=code_path,
            )
        except Exception as e:
            self._log(f"Failed to load TOTEM model: {e}")
            raise

        device = config.device
        compression_factor = config.compression_factor
        self._log(f"TOTEM model loaded (device={device}, compression_factor={compression_factor})")

        # Get hyperparameters
        window_size = int(self.params.get("window_size", 96))
        normalization = self.params.get("normalization", "zscore")
        merge_mode = self.params.get("merge_mode", "non_overlap")
        if merge_mode == "overlap":
            stride = int(self.params.get("stride", max(1, window_size // 2)))
        else:
            # Non-overlap mode is fixed by definition.
            stride = window_size
        if stride < 1:
            raise ValueError("stride must be >= 1")
        preview_updates = int(self.params.get("preview_updates", 10))

        self._log(
            f"Imputing {len(valid_columns)} column(s) with window_size={window_size}, "
            f"merge_mode={merge_mode}, stride={stride}"
        )

        # Calculate total steps for progress
        total_columns = len(valid_columns)
        preview_stride = max(1, total_columns // max(1, preview_updates))

        # Process each column independently
        for col_idx, col in enumerate(valid_columns):
            self._log(f"Imputing column: {col}")

            imputed_values = self._impute_column(
                df=df,
                column=col,
                model=model,
                compression_factor=compression_factor,
                window_size=window_size,
                stride=stride,
                normalization=normalization,
                device=device,
            )

            # Apply imputed values only to missing positions
            missing_mask = df[col].isna()
            result.loc[missing_mask, col] = imputed_values[missing_mask]

            # Progress callback
            if self.progress_callback:
                progress = (col_idx + 1) / total_columns
                self.progress_callback(col_idx + 1, total_columns, 0.0, None)

            # Impute preview callback
            if self.impute_callback and (col_idx % preview_stride == 0 or col_idx == total_columns - 1):
                self.impute_callback(col_idx + 1, total_columns, result.copy())

        self._log("TOTEM imputation completed")
        return result

    def _impute_column(
        self,
        df: pd.DataFrame,
        column: str,
        model: Any,
        compression_factor: int,
        window_size: int,
        stride: int,
        normalization: str,
        device: str,
    ) -> np.ndarray:
        """Impute a single column using TOTEM VQVAE.

        Args:
            df: Input dataframe
            column: Column name to impute
            model: Loaded VQVAE model
            compression_factor: VQVAE compression factor
            window_size: Window size for processing (must be divisible by compression_factor)
            stride: Sliding stride. For non-overlap this equals window_size.
            normalization: Normalization method ("zscore" or "minmax")
            device: Torch device

        Returns:
            Array of imputed values (same length as df)
        """
        data = df[column].values.astype(np.float64)
        n = len(data)
        mask = ~np.isnan(data)  # True = observed

        # Initialize result with original data
        result = data.copy()

        # Fill NaN with 0 for model input (will be replaced with predictions)
        data_filled = np.nan_to_num(data, nan=0.0)

        # Apply normalization
        if normalization == "zscore":
            data_norm, norm_param1, norm_param2 = apply_zscore_normalization(data_filled, mask)
            inverse_fn = inverse_zscore
        else:
            data_norm, norm_param1, norm_param2 = apply_minmax_normalization(data_filled, mask)
            inverse_fn = inverse_minmax

        # TOTEM masking convention: missing inputs are explicitly set to 0.
        data_norm[~mask] = 0.0

        # Collect predictions on missing points. Overlap mode averages duplicates.
        pred_sum = np.zeros(n, dtype=np.float64)
        pred_count = np.zeros(n, dtype=np.int32)

        if n <= window_size:
            num_windows = 1
        else:
            num_windows = ((n - window_size) // stride) + 1
            if (n - window_size) % stride != 0:
                num_windows += 1

        for w_idx in range(num_windows):
            start = w_idx * stride
            if start >= n:
                break
            end = min(start + window_size, n)
            actual_len = end - start

            # Skip if no missing values in this window
            window_mask = mask[start:end]
            if window_mask.all():
                continue

            # Pad window to window_size if needed
            window_data = np.zeros(window_size, dtype=np.float64)
            window_data[:actual_len] = data_norm[start:end]

            # Convert to tensor: [1, 1, window_size] (batch=1, nvars=1)
            x_tensor = torch.tensor(
                window_data.reshape(1, 1, window_size),
                dtype=torch.float32,
                device=device,
            )

            # Encode
            codes, code_ids, codebook = revintime2codes(
                x_tensor,
                compression_factor,
                model.encoder,
                model.vq,
            )

            # Decode
            predictions = codes2timerevin(
                code_ids,
                codebook,
                compression_factor,
                model.decoder,
            )

            # Extract predictions: [1, window_size, 1] -> [window_size]
            pred_np = predictions.cpu().numpy().squeeze()

            # Inverse normalization
            pred_unnorm = inverse_fn(pred_np, norm_param1, norm_param2)

            # Collect predictions only for missing positions in this window.
            for i in range(actual_len):
                if not mask[start + i]:
                    idx = start + i
                    pred_sum[idx] += pred_unnorm[i]
                    pred_count[idx] += 1

        missing_idx = ~mask
        valid_pred = pred_count > 0
        use_pred = missing_idx & valid_pred
        result[use_pred] = pred_sum[use_pred] / pred_count[use_pred]

        return result
