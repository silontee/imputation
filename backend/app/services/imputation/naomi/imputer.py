"""NAOMI imputer – BaseImputer interface wrapping train + impute."""

from typing import Any, Dict, List, Optional, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from app.services.imputation.base import BaseImputer
from app.services.imputation.naomi.trainer import train_naomi


class NAOMIImputer(BaseImputer):
    """NAOMI deep-learning sequence imputer.

    Params (via hyperparameters dict):
        hidden_dim: int (default 64)
        epochs: int (default 50)
        lr: float (default 1e-3)
        num_resolutions: int | None (auto if None)
        highest: int | None (override max step size)
        window_size: int (default 50)
        batch_size: int (default 64)
        n_layers: int (default 2)
        clip: float (default 10.0)
        preview_updates: int (default 10) - number of intermediate previews
        preview_stride: int | None (override stride)
    """

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        progress_callback: Optional[Callable] = None,
        impute_callback: Optional[Callable] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(params)
        self.progress_callback = progress_callback
        self.impute_callback = impute_callback
        self.log_callback = log_callback

    @property
    def name(self) -> str:
        return "NAOMI"

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns_to_impute: List[str],
        column_types: Dict[str, str],
        feature_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        result = df.copy()

        if not columns_to_impute:
            return result

        # Determine columns to use for the model
        if feature_columns is not None:
            model_cols = list(dict.fromkeys(feature_columns + columns_to_impute))
        else:
            # Use all numeric-compatible columns
            model_cols = columns_to_impute[:]
            for col in df.columns:
                if col not in model_cols:
                    ctype = column_types.get(col, "")
                    if ctype == "NUMERIC" or pd.api.types.is_numeric_dtype(df[col]):
                        model_cols.append(col)

        # Filter to columns that exist
        model_cols = [c for c in model_cols if c in df.columns]
        if not model_cols:
            return result

        # Prepare numeric matrix
        numeric_df, encoders, datetime_cols = self._prepare_columns(
            df, model_cols, column_types
        )
        if numeric_df.shape[1] == 0:
            return result

        col_order = numeric_df.columns.tolist()
        data_np = numeric_df.values.astype(np.float64)

        # Build mask (1 = observed, 0 = missing)
        mask_full = (~np.isnan(data_np)).astype(np.float64)
        # Collapse to single channel: missing if *any* target column is missing
        target_indices = [col_order.index(c) for c in columns_to_impute if c in col_order]
        mask_1d = np.ones(len(data_np), dtype=np.float64)
        for idx in target_indices:
            mask_1d *= mask_full[:, idx]
        mask_1d = mask_1d.reshape(-1, 1)

        # Scale (notebook uses MinMax)
        scaler = MinMaxScaler()
        data_filled = np.nan_to_num(data_np, nan=0.0)
        data_scaled = scaler.fit_transform(data_filled)

        # Hyperparams
        hidden_dim = int(self.params.get("hidden_dim", 64))
        epochs = int(self.params.get("epochs", 50))
        lr = float(self.params.get("lr", 1e-3))
        num_res = self.params.get("num_resolutions", None)
        if num_res is not None:
            num_res = int(num_res)
        highest = self.params.get("highest", None)
        if highest is not None:
            highest = int(highest)
        # Device selection: CUDA → MPS → CPU
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        import logging
        logging.getLogger(__name__).info(f"NAOMI using device: {device}")

        x_tensor = torch.tensor(data_scaled, dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(mask_1d, dtype=torch.float32, device=device)

        preview_updates = int(self.params.get("preview_updates", 10))
        train_stride = max(1, epochs // max(1, preview_updates))

        def _train_progress(epoch: int, total_epochs: int, loss_val: float, model) -> None:
            if self.progress_callback is not None:
                self.progress_callback(epoch, total_epochs, loss_val)
            if self.impute_callback is None:
                return
            if epoch % train_stride != 0 and epoch != total_epochs:
                return
            import torch
            with torch.no_grad():
                model.eval()
                x_imputed = model.impute(x_tensor, mask_tensor)
                model.train()
            imputed_np = x_imputed.detach().cpu().numpy()
            imputed_unscaled = scaler.inverse_transform(imputed_np)
            preview_df = self._apply_imputed_array(
                df,
                columns_to_impute,
                column_types,
                col_order,
                imputed_unscaled,
                encoders,
                datetime_cols,
            )
            self.impute_callback(epoch, total_epochs, preview_df)

        # Row completeness for training windows
        row_complete = ~np.isnan(data_np).any(axis=1)

        # Adjust window_size based on contiguous fully observed rows
        window_size = int(self.params.get("window_size", 50))
        max_run = 0
        run = 0
        for ok in row_complete:
            if ok:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        if max_run >= 2 and max_run < window_size:
            if self.log_callback is not None:
                self.log_callback(
                    f"NAOMI: reduced window_size from {window_size} to {max_run} due to limited complete rows"
                )
            window_size = max_run
        elif max_run < 2:
            if self.log_callback is not None:
                self.log_callback(
                    "NAOMI: no fully observed windows; training on filled data (fallback)"
                )

        model = train_naomi(
            data=data_scaled,
            row_complete_mask=row_complete,
            hidden_dim=hidden_dim,
            num_resolutions=num_res,
            highest=highest,
            epochs=epochs,
            lr=lr,
            window_size=window_size,
            batch_size=int(self.params.get("batch_size", 64)),
            n_layers=int(self.params.get("n_layers", 2)),
            clip=float(self.params.get("clip", 10.0)),
            device=device,
            progress_callback=_train_progress,
        )

        # Impute
        preview_stride = self.params.get("preview_stride", None)
        stride = None

        def _emit_preview(step: int, total: int, x_scaled_step, filled_mask=None) -> None:
            nonlocal stride
            if self.impute_callback is None:
                return
            if stride is None:
                if preview_stride is not None:
                    try:
                        stride = max(1, int(preview_stride))
                    except Exception:
                        stride = 1
                else:
                    stride = max(1, total // max(1, preview_updates))

            if step % stride != 0 and step != total:
                return

            imputed_step = x_scaled_step.detach().cpu().numpy()
            imputed_unscaled = scaler.inverse_transform(imputed_step)
            preview_df = self._apply_imputed_array(
                df,
                columns_to_impute,
                column_types,
                col_order,
                imputed_unscaled,
                encoders,
                datetime_cols,
                filled_mask=filled_mask,
            )
            self.impute_callback(step, total, preview_df)

        x_imputed = model.impute(
            x_tensor,
            mask_tensor,
            progress_callback=_emit_preview if self.impute_callback else None,
        )
        imputed_np = x_imputed.cpu().numpy()

        # Inverse scale
        imputed_unscaled = scaler.inverse_transform(imputed_np)

        result = self._apply_imputed_array(
            df,
            columns_to_impute,
            column_types,
            col_order,
            imputed_unscaled,
            encoders,
            datetime_cols,
        )

        return result

    # ---- column preparation (similar to KNN imputer) ----------------------

    @staticmethod
    def _prepare_columns(
        df: pd.DataFrame,
        model_cols: List[str],
        column_types: Dict[str, str],
    ):
        """Convert selected columns to numeric. Returns (numeric_df, encoders, datetime_cols)."""
        numeric_df = pd.DataFrame(index=df.index)
        encoders: Dict[str, LabelEncoder] = {}
        datetime_cols: List[str] = []

        for col in model_cols:
            if col not in df.columns:
                continue
            col_type = column_types.get(col, "")

            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_df[col] = df[col].astype(float)
                continue

            if col_type == "DATETIME" or df[col].dtype == "datetime64[ns]":
                try:
                    dt = pd.to_datetime(df[col], errors="coerce")
                    numeric_df[col] = dt.astype("int64") / 1e9
                    datetime_cols.append(col)
                except Exception:
                    pass
                continue

            # Categorical / string
            if col_type in ("CATEGORICAL", "") or df[col].dtype == object:
                le = LabelEncoder()
                non_null = df[col].notna()
                if non_null.any():
                    le.fit(df.loc[non_null, col].astype(str))
                    numeric_df[col] = np.nan
                    numeric_df.loc[non_null, col] = le.transform(
                        df.loc[non_null, col].astype(str)
                    ).astype(float)
                    encoders[col] = le

        return numeric_df, encoders, datetime_cols

    @staticmethod
    def _apply_imputed_array(
        df: pd.DataFrame,
        columns_to_impute: List[str],
        column_types: Dict[str, str],
        col_order: List[str],
        imputed_unscaled: np.ndarray,
        encoders: Dict[str, LabelEncoder],
        datetime_cols: List[str],
        filled_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Apply imputed values to a copy of df for target columns only."""
        result = df.copy()
        for col_name in columns_to_impute:
            if col_name not in col_order:
                continue
            ci = col_order.index(col_name)
            missing_rows = df[col_name].isna().to_numpy()
            if filled_mask is not None:
                missing_rows = missing_rows & filled_mask.astype(bool)
            if not missing_rows.any():
                continue

            values = imputed_unscaled[missing_rows, ci]

            if col_name in encoders:
                le = encoders[col_name]
                codes = np.clip(np.round(values).astype(int), 0, len(le.classes_) - 1)
                result.loc[missing_rows, col_name] = le.inverse_transform(codes)
            elif col_name in datetime_cols:
                result.loc[missing_rows, col_name] = pd.to_datetime(
                    values * 1e9, unit="ns", errors="coerce"
                )
            else:
                # Default numeric
                result.loc[missing_rows, col_name] = values

        return result
