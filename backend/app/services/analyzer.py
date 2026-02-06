from typing import Any, List, Tuple
import pandas as pd
import numpy as np

from app.schemas.analyze import ColumnProfile, ColumnType, ColumnAction


def _safe_examples(series: pd.Series, k: int = 3) -> List[Any]:
    """Get k example values from series, handling nulls."""
    vals = series.dropna().unique().tolist()
    # Convert numpy types to Python types
    result = []
    for v in vals[:k]:
        if isinstance(v, (np.integer, np.floating)):
            result.append(float(v) if isinstance(v, np.floating) else int(v))
        else:
            result.append(str(v) if v is not None else None)
    return result


def infer_column_type(series: pd.Series) -> Tuple[ColumnType, List[str]]:
    """Infer column type from pandas Series."""
    warnings: List[str] = []
    s = series.dropna()

    # Empty -> default categorical
    if len(s) == 0:
        return ColumnType.CATEGORICAL, ["All values are null in sample"]

    # Check if it's a datetime
    try:
        dt = pd.to_datetime(s, errors="coerce", format="mixed")
        dt_success = dt.notna().mean()
        if dt_success >= 0.90:
            return ColumnType.DATETIME, warnings
    except Exception:
        pass

    # Check if it's numeric (allow numeric strings)
    num = pd.to_numeric(s.astype(str), errors="coerce")
    num_success = num.notna().mean()
    if num_success >= 0.95:
        # Low cardinality numeric might be categorical
        if s.nunique() <= 20:
            warnings.append("Numeric but low cardinality - might be categorical")
            return ColumnType.CATEGORICAL, warnings
        return ColumnType.NUMERIC, warnings

    # Check if it's an ID (all unique values)
    if s.nunique() == len(s):
        return ColumnType.ID, ["All values unique in sample - likely ID column"]

    return ColumnType.CATEGORICAL, warnings


def recommend_action(col_type: ColumnType, null_ratio: float) -> ColumnAction:
    """Recommend action based on column type and null ratio."""
    if col_type == ColumnType.ID:
        return ColumnAction.IGNORE
    # No missing: default ignore
    if null_ratio == 0:
        return ColumnAction.IGNORE
    return ColumnAction.IMPUTE


def analyze_dataframe(df: pd.DataFrame) -> List[ColumnProfile]:
    """Analyze dataframe and return column profiles."""
    profiles: List[ColumnProfile] = []
    n = len(df)

    for name in df.columns:
        series = df[name]
        null_count = int(series.isna().sum())
        null_ratio = float(null_count / max(n, 1))
        unique_count = int(series.dropna().nunique())
        col_type, warnings = infer_column_type(series)

        profiles.append(
            ColumnProfile(
                name=str(name),
                detected_type=col_type,
                null_count=null_count,
                null_ratio=round(null_ratio, 6),
                unique_count=unique_count,
                example=_safe_examples(series),
                recommended_action=recommend_action(col_type, null_ratio),
                warnings=warnings,
            )
        )

    return profiles
