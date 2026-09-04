"""AIML: a small, modular toolkit for data preprocessing, machine learning and deep learning."""

from aiml.preprocessing import (
    Preprocessor,
    clean_dataframe,
    remove_outliers,
    split_data,
    split_features_target,
)

__version__ = "0.1.0"

__all__ = [
    "Preprocessor",
    "clean_dataframe",
    "remove_outliers",
    "split_data",
    "split_features_target",
    "__version__",
]
