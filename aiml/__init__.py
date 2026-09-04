"""AIML: a small, modular toolkit for data preprocessing, machine learning and deep learning."""

from aiml.preprocessing import (
    Preprocessor,
    clean_dataframe,
    remove_outliers,
    split_data,
    split_features_target,
)

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy access to the optional deep learning models, e.g. ``aiml.MLPClassifier``.
    if name in {"MLPClassifier", "MLPRegressor", "CNNClassifier"}:
        from aiml import deep

        return getattr(deep, name)
    raise AttributeError(f"module 'aiml' has no attribute {name!r}")


__all__ = [
    "Preprocessor",
    "clean_dataframe",
    "remove_outliers",
    "split_data",
    "split_features_target",
    "MLPClassifier",
    "MLPRegressor",
    "CNNClassifier",
    "__version__",
]
