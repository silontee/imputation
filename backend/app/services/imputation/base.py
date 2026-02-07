from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd


class BaseImputer(ABC):
    """Base class for imputation methods."""

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}

    @abstractmethod
    def fit_transform(
        self,
        df: pd.DataFrame,
        columns_to_impute: List[str],
        column_types: Dict[str, str],
        feature_columns: List[str] | None = None,
    ) -> pd.DataFrame:
        """Fit and transform the dataframe.

        Args:
            df: Full dataframe.
            columns_to_impute: TARGET columns whose missing values will be filled.
            column_types: Mapping of column name -> type string.
            feature_columns: FEATURE columns used as model input only.
                If None, all columns in df are used as features.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the imputation method."""
        pass


class JobCanceledError(Exception):
    """Raised when a job is canceled during processing."""
