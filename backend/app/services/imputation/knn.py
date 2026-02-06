from typing import Any, Dict, List
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

from app.services.imputation.base import BaseImputer


class KNNMethodImputer(BaseImputer):
    """KNN-based imputation.
    
    Uses all available columns as features for imputation:
    - Numeric columns: used directly
    - DateTime columns: converted to timestamp
    - Categorical columns: label encoded
    """

    @property
    def name(self) -> str:
        return "KNN"

    def _prepare_all_columns(
        self, df: pd.DataFrame, column_types: Dict[str, str]
    ) -> tuple[pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
        """Convert all columns to numeric for KNN.
        
        Returns:
            - numeric_df: DataFrame with all columns as numeric
            - encoders: dict of LabelEncoder for categorical columns
            - datetime_cols: list of datetime column names (for restoration)
        """
        numeric_df = df.copy()
        encoders = {}
        datetime_cols = []
        
        for col in df.columns:
            col_type = column_types.get(col, "")
            
            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            # Handle DateTime columns
            if col_type == "DATETIME" or df[col].dtype == 'datetime64[ns]':
                try:
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                    # Convert to timestamp (seconds since epoch)
                    numeric_df[col] = dt_series.astype('int64') / 1e9
                    datetime_cols.append(col)
                except Exception:
                    # If conversion fails, drop the column for KNN
                    numeric_df = numeric_df.drop(columns=[col])
                continue
            
            # Handle Categorical/String columns with LabelEncoder
            if col_type in ["CATEGORICAL", ""] or df[col].dtype == 'object':
                le = LabelEncoder()
                non_null_mask = df[col].notna()
                if non_null_mask.any():
                    # Fit on non-null values
                    le.fit(df.loc[non_null_mask, col].astype(str))
                    # Transform non-null values
                    numeric_df.loc[non_null_mask, col] = le.transform(
                        df.loc[non_null_mask, col].astype(str)
                    )
                    # NaN values stay as NaN
                    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                    encoders[col] = le
                else:
                    # All NaN column - drop for KNN
                    numeric_df = numeric_df.drop(columns=[col])
        
        return numeric_df, encoders, datetime_cols

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns_to_impute: List[str],
        column_types: Dict[str, str],
        feature_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        result = df.copy()
        n_neighbors = self.params.get("n_neighbors", 5)

        # Prepare all columns for KNN (convert to numeric)
        numeric_df, encoders, datetime_cols = self._prepare_all_columns(df, column_types)

        # Get columns that actually exist in numeric_df
        available_cols = [col for col in columns_to_impute if col in numeric_df.columns]
        if not available_cols:
            return result

        # Build input column set: feature + target (or all if no features specified)
        if feature_columns is not None:
            valid_features = [c for c in feature_columns if c in numeric_df.columns]
            all_cols = list(dict.fromkeys(valid_features + available_cols))
        else:
            all_cols = numeric_df.columns.tolist()
        
        # Run KNN on all numeric columns
        try:
            # Ensure n_neighbors doesn't exceed sample count
            effective_neighbors = min(n_neighbors, len(numeric_df) - 1)
            effective_neighbors = max(1, effective_neighbors)
            
            imputer = KNNImputer(
                n_neighbors=effective_neighbors,
                weights=self.params.get("weights", "uniform"),
                metric=self.params.get("metric", "nan_euclidean"),
            )
            imputed_values = imputer.fit_transform(numeric_df[all_cols])
            imputed_df = pd.DataFrame(
                imputed_values,
                columns=all_cols,
                index=numeric_df.index
            )
        except Exception as e:
            # If KNN fails, return original
            print(f"KNN failed: {e}")
            return result
        
        # Apply imputed values only to target columns
        for col in columns_to_impute:
            if col not in imputed_df.columns:
                continue
                
            col_type = column_types.get(col, "")
            
            if col_type == "NUMERIC" or pd.api.types.is_numeric_dtype(df[col]):
                # Numeric: use imputed values directly
                result[col] = imputed_df[col]
                
            elif col in datetime_cols:
                # DateTime: convert back from timestamp
                result[col] = pd.to_datetime(imputed_df[col] * 1e9, unit='ns', errors='coerce')
                
            elif col in encoders:
                # Categorical: inverse transform
                imputed_codes = imputed_df[col].round().astype(int)
                imputed_codes = imputed_codes.clip(0, len(encoders[col].classes_) - 1)
                result[col] = encoders[col].inverse_transform(imputed_codes)
        
        return result
