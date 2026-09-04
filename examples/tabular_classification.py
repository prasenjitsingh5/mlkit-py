"""End-to-end tabular classification on the Wisconsin breast cancer dataset.

Run from the repository root:

    python examples/tabular_classification.py

The dataset ships with scikit-learn, so nothing is downloaded. The script
cleans the data, removes outliers, splits it, fits the Preprocessor, trains
an MLPClassifier with early stopping, and prints a classification report.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

from mlkit import Preprocessor, clean_dataframe, remove_outliers, split_data
from mlkit.deep import MLPClassifier


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "malignant"})
    # In this dataset 0 is malignant, 1 is benign. Flip so 1 means malignant.
    df["malignant"] = 1 - df["malignant"]
    print(f"rows: {len(df)}, features: {df.shape[1] - 1}")

    df = clean_dataframe(df)
    df = remove_outliers(df, method="zscore", factor=4.0)
    print(f"rows after outlier removal: {len(df)}")

    X_train, X_test, y_train, y_test = split_data(df, target="malignant", test_size=0.25, stratify=True)

    prep = Preprocessor(scaler="standard").fit(X_train)
    X_train_p = prep.transform(X_train)
    X_test_p = prep.transform(X_test)

    clf = MLPClassifier(
        hidden_sizes=(64, 32),
        dropout=0.1,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        validation_fraction=0.2,
        early_stopping_patience=10,
        random_state=0,
    )
    clf.fit(X_train_p, y_train)
    print(f"stopped after {clf.n_epochs_} epochs")

    y_pred = clf.predict(X_test_p)
    print(f"test accuracy: {clf.score(X_test_p, y_test):.3f}\n")
    print(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    main()
