from typing import Any, Dict, List
import pandas as pd
import numpy as np

from app.services.imputation.base import BaseImputer


class MeanModeImputer(BaseImputer):
    """Simple imputer using mean for numeric and mode for categorical."""

    @property
    def name(self) -> str:
        return "MEAN"

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns_to_impute: List[str],
        column_types: Dict[str, str],
        feature_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        result = df.copy()

        for col in columns_to_impute:
            if col not in result.columns:
                continue

            col_type = column_types.get(col, "CATEGORICAL")

            if col_type == "NUMERIC":
                # Use mean for numeric columns
                mean_val = result[col].mean()
                if pd.notna(mean_val):
                    result[col] = result[col].fillna(mean_val)
            elif col_type == "DATETIME":
                # Use median for datetime
                non_null = result[col].dropna()
                if len(non_null) > 0:
                    # Convert to numeric, get median, convert back
                    numeric_times = pd.to_numeric(pd.to_datetime(non_null))
                    median_time = pd.to_datetime(np.median(numeric_times))
                    result[col] = result[col].fillna(median_time)
            else:
                # Use mode for categorical columns
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val.iloc[0])

        return result
