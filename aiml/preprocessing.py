"""Data cleaning and preprocessing utilities.

The module has two layers:

* Plain functions that operate on a :class:`pandas.DataFrame` and return a
  new frame (``clean_dataframe``, ``remove_outliers``, ``split_data``).
* :class:`Preprocessor`, a scikit-learn compatible transformer that imputes,
  scales and one-hot encodes a mixed numeric / categorical frame so it can be
  fed to any model.

Example
-------
>>> import pandas as pd
>>> from aiml.preprocessing import Preprocessor, clean_dataframe, split_data
>>> df = pd.DataFrame({
...     "age": [23, 45, None, 31],
...     "city": ["Delhi", "Pune", "Delhi", None],
...     "bought": [0, 1, 0, 1],
... })
>>> df = clean_dataframe(df)
>>> X_train, X_test, y_train, y_test = split_data(df, target="bought", test_size=0.5)
>>> prep = Preprocessor().fit(X_train)
>>> prep.transform(X_test).shape[0] == len(X_test)
True
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

__all__ = [
    "Preprocessor",
    "clean_dataframe",
    "remove_outliers",
    "split_data",
    "split_features_target",
]


# --------------------------------------------------------------------------- #
# Plain DataFrame helpers
# --------------------------------------------------------------------------- #
def clean_dataframe(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    strip_strings: bool = True,
    empty_as_na: bool = True,
    drop_na_columns_threshold: float | None = None,
    drop_na_rows: bool = False,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a cleaned copy of ``df``.

    Parameters
    ----------
    df:
        Input frame. It is never modified in place.
    drop_duplicates:
        Drop exact duplicate rows.
    strip_strings:
        Strip leading and trailing whitespace from string columns.
    empty_as_na:
        Treat empty strings (after stripping) as missing values.
    drop_na_columns_threshold:
        If given, drop any column whose fraction of missing values is
        strictly greater than this number (between 0 and 1).
    drop_na_rows:
        Drop rows that still contain any missing value after the steps above.
    columns:
        Restrict the output to these columns, in this order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataframe expects a pandas DataFrame")
    if drop_na_columns_threshold is not None and not 0 <= drop_na_columns_threshold <= 1:
        raise ValueError("drop_na_columns_threshold must be between 0 and 1")

    out = df.copy()
    if columns is not None:
        out = out.loc[:, list(columns)]

    if strip_strings or empty_as_na:
        for col in out.columns:
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
                series = out[col].astype(object)
                is_str = series.map(lambda v: isinstance(v, str))
                if strip_strings:
                    series = series.where(~is_str, series.where(~is_str, series.astype(str).str.strip()))
                if empty_as_na:
                    series = series.mask(is_str & (series == ""), np.nan)
                out[col] = series.where(series.notna(), np.nan)

    if drop_duplicates:
        out = out.drop_duplicates()

    if drop_na_columns_threshold is not None and len(out):
        na_fraction = out.isna().mean()
        out = out.loc[:, na_fraction <= drop_na_columns_threshold]

    if drop_na_rows:
        out = out.dropna()

    return out.reset_index(drop=True)


def remove_outliers(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """Drop rows whose value in any of ``columns`` is an outlier.

    Parameters
    ----------
    columns:
        Numeric columns to inspect. Defaults to every numeric column.
    method:
        ``"iqr"`` keeps values within ``[Q1 - factor*IQR, Q3 + factor*IQR]``.
        ``"zscore"`` keeps values whose absolute z-score is at most ``factor``
        (a sensible ``factor`` for z-score is 3.0).
    factor:
        Width of the accepted band. See ``method``.
    """
    if method not in {"iqr", "zscore"}:
        raise ValueError("method must be 'iqr' or 'zscore'")
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    columns = list(columns)
    if not columns:
        return df.copy()

    keep = pd.Series(True, index=df.index)
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            mask = series.between(lower, upper)
        else:
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                mask = pd.Series(True, index=df.index)
            else:
                mask = ((series - series.mean()).abs() / std) <= factor
        # Missing values are not outliers; leave them for the imputer.
        keep &= mask | series.isna()

    return df.loc[keep].reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` where ``y`` is the ``target`` column and ``X`` is the rest."""
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in DataFrame")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_data(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int | None = 42,
    shuffle: bool = True,
):
    """Split a frame into ``X_train, X_test, y_train, y_test``.

    Set ``stratify=True`` for classification targets to keep class balance in
    both halves.
    """
    X, y = split_features_target(df, target)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y if stratify else None,
    )


# --------------------------------------------------------------------------- #
# Scikit-learn compatible transformer
# --------------------------------------------------------------------------- #
class Preprocessor(BaseEstimator, TransformerMixin):
    """Impute, scale and encode a mixed-type DataFrame.

    Numeric columns are imputed with ``numeric_impute`` and then scaled.
    Categorical columns are imputed with the most frequent value and one-hot
    encoded. Unknown categories seen at transform time are ignored, so the
    output width is fixed after ``fit``.

    Parameters
    ----------
    numeric_columns, categorical_columns:
        Explicit column lists. When ``None`` they are inferred from dtypes at
        ``fit`` time: numeric dtypes are numeric, everything else is
        categorical.
    scaler:
        ``"standard"``, ``"minmax"`` or ``None`` for no scaling.
    numeric_impute:
        ``"median"``, ``"mean"`` or ``"most_frequent"``.
    drop_columns:
        Columns to discard before processing (ids, free text, leakage).
    return_dataframe:
        Return a :class:`pandas.DataFrame` with readable column names instead
        of a NumPy array.
    """

    def __init__(
        self,
        numeric_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
        *,
        scaler: str | None = "standard",
        numeric_impute: str = "median",
        drop_columns: Sequence[str] | None = None,
        return_dataframe: bool = True,
    ):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.scaler = scaler
        self.numeric_impute = numeric_impute
        self.drop_columns = drop_columns
        self.return_dataframe = return_dataframe

    # ----------------------------------------------------------------- utils
    def _make_scaler(self):
        if self.scaler is None or self.scaler == "none":
            return "passthrough"
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "minmax":
            return MinMaxScaler()
        raise ValueError("scaler must be 'standard', 'minmax' or None")

    def _resolve_columns(self, X: pd.DataFrame):
        dropped = set(self.drop_columns or [])
        remaining = [c for c in X.columns if c not in dropped]
        if self.numeric_columns is None:
            numeric = [
                c
                for c in remaining
                if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
            ]
        else:
            numeric = [c for c in self.numeric_columns if c not in dropped]
        if self.categorical_columns is None:
            categorical = [c for c in remaining if c not in numeric]
        else:
            categorical = [c for c in self.categorical_columns if c not in dropped]
        return numeric, categorical

    @staticmethod
    def _as_object(X: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        X = X.copy()
        for c in columns:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        return X

    # ------------------------------------------------------------- sklearn API
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessor expects a pandas DataFrame")
        if self.numeric_impute not in {"median", "mean", "most_frequent"}:
            raise ValueError("numeric_impute must be 'median', 'mean' or 'most_frequent'")

        numeric, categorical = self._resolve_columns(X)
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical

        transformers = []
        if numeric:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy=self.numeric_impute)),
                            ("scale", self._make_scaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    categorical,
                )
            )
        if not transformers:
            raise ValueError("no columns left to preprocess")

        self.column_transformer_ = ColumnTransformer(transformers, remainder="drop")
        self.column_transformer_.fit(self._as_object(X, categorical))
        self.feature_names_out_ = self._feature_names()
        return self

    def transform(self, X: pd.DataFrame):
        if not hasattr(self, "column_transformer_"):
            raise RuntimeError("Preprocessor is not fitted; call fit first")
        missing = [c for c in self.numeric_columns_ + self.categorical_columns_ if c not in X.columns]
        if missing:
            raise KeyError(f"columns missing at transform time: {missing}")
        arr = self.column_transformer_.transform(self._as_object(X, self.categorical_columns_))
        if self.return_dataframe:
            return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)
        return arr

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Preprocessor is not fitted; call fit first")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _feature_names(self) -> list[str]:
        names: list[str] = list(self.numeric_columns_)
        if self.categorical_columns_:
            onehot = self.column_transformer_.named_transformers_["cat"].named_steps["onehot"]
            for col, cats in zip(self.categorical_columns_, onehot.categories_):
                names.extend(f"{col}={cat}" for cat in cats)
        return names
