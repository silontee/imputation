from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import IO, Tuple, Union

import pandas as pd
import numpy as np


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]


SourceType = Union[str, Path, bytes, bytearray, IO[bytes]]


def read_csv_with_fallback(
    source: SourceType,
    **kwargs,
) -> Tuple[pd.DataFrame, str]:
    """Read CSV with encoding fallback.

    Returns:
        (df, encoding_used)
    """
    # Normalize to bytes for file-like sources
    if hasattr(source, "read") and not isinstance(source, (bytes, bytearray)):
        source = source.read()

    last_err: Exception | None = None
    for enc in ENCODING_CANDIDATES:
        try:
            if isinstance(source, (bytes, bytearray)):
                buf = BytesIO(source)
                df = pd.read_csv(buf, encoding=enc, **kwargs)
            else:
                df = pd.read_csv(source, encoding=enc, **kwargs)
            df = normalize_missing_values(df)
            return df, enc
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except ValueError as e:
            msg = str(e)
            if "codec can't decode" in msg or "codec can't encode" in msg:
                last_err = e
                continue
            raise

    if last_err is not None:
        raise last_err

    # Fallback to pandas default (should rarely reach here)
    df = pd.read_csv(source, **kwargs)
    df = normalize_missing_values(df)
    return df, "unknown"


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Treat 0 and 'NA' (case-insensitive) as missing values."""
    out = df.copy()

    # String/object columns: treat 'NA' as missing
    obj_cols = out.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        s = out[col].astype(str).str.strip()
        mask_na = s.str.upper().eq("NA")
        if mask_na.any():
            out.loc[mask_na, col] = np.nan

    # Numeric columns: treat 0 as missing
    num_cols = out.select_dtypes(include=["number"]).columns
    for col in num_cols:
        out.loc[out[col] == 0, col] = np.nan

    return out
