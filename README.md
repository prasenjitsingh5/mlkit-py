# mlkit-py

[![tests](https://github.com/prasenjitsingh5/mlkit-py/actions/workflows/tests.yml/badge.svg)](https://github.com/prasenjitsingh5/mlkit-py/actions/workflows/tests.yml)

A small, modular Python toolkit for data preprocessing, machine learning and deep learning. It gives you clean, scikit-learn compatible building blocks so you can go from a raw CSV to a trained model in a few lines.

---

## Features

- **Data preprocessing**: clean frames, drop outliers, split data, and a `Preprocessor` transformer that imputes, scales and one-hot encodes mixed numeric and categorical columns.
- **Deep learning models**: `MLPClassifier`, `MLPRegressor` and `CNNClassifier` built on PyTorch with a `fit` / `predict` API, early stopping, GPU auto-detection and save / load.
- **Works with scikit-learn**: every estimator drops into a `Pipeline`, `GridSearchCV` or `cross_val_score`.
- **Optional heavy dependencies**: PyTorch is only needed for `mlkit.deep`.

---

## Installation

```bash
pip install git+https://github.com/prasenjitsingh5/mlkit-py.git
```

With the deep learning extra:

```bash
pip install "mlkit-py[deep] @ git+https://github.com/prasenjitsingh5/mlkit-py.git"
```

For local development with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/prasenjitsingh5/mlkit-py.git
cd mlkit-py
uv sync --extra dev --extra deep
uv run pytest
```

Requires Python 3.10 or newer. The committed `uv.lock` pins every dependency, and PyTorch resolves from the CPU-only index so installs stay small. Use `uv sync --extra dev` alone to skip PyTorch; the deep learning tests are skipped automatically.

---

## Quick start

A runnable version of this on a public dataset is in [examples/tabular_classification.py](examples/tabular_classification.py).

```python
import pandas as pd
from mlkit import Preprocessor, clean_dataframe, remove_outliers, split_data
from mlkit.deep import MLPClassifier

df = pd.read_csv("customers.csv")

# 1. Clean: strip whitespace, treat "" as missing, drop duplicates
df = clean_dataframe(df)

# 2. Remove extreme rows in numeric columns (IQR rule)
df = remove_outliers(df, ["age", "income"])

# 3. Split into train / test, keeping class balance
X_train, X_test, y_train, y_test = split_data(df, target="churned", stratify=True)

# 4. Impute, scale and one-hot encode
prep = Preprocessor().fit(X_train)
X_train_p = prep.transform(X_train)
X_test_p = prep.transform(X_test)

# 5. Train a neural network and evaluate
clf = MLPClassifier(hidden_sizes=(64, 32), epochs=50, random_state=0)
clf.fit(X_train_p, y_train)
print("accuracy:", clf.score(X_test_p, y_test))
```

---

## Usage examples

### Cleaning a DataFrame

```python
from mlkit import clean_dataframe

df = clean_dataframe(
    raw,
    drop_duplicates=True,  # exact duplicate rows
    strip_strings=True,  # "  Delhi " -> "Delhi"
    empty_as_na=True,  # "" -> NaN
    drop_na_columns_threshold=0.6,  # drop columns that are >60% missing
    drop_na_rows=False,  # keep rows with NaN for the imputer
)
```

`clean_dataframe` never modifies the input. It returns a new frame with a fresh index.

### Removing outliers

```python
from mlkit import remove_outliers

# Inter-quartile rule, 1.5 * IQR (default)
tidy = remove_outliers(df, ["price", "quantity"])

# Z-score rule, keep |z| <= 3
tidy = remove_outliers(df, ["price"], method="zscore", factor=3.0)
```

Rows with a missing value in the checked column are kept so the imputer can handle them later.

### Preprocessor

```python
from mlkit import Preprocessor

prep = Preprocessor(
    scaler="minmax",  # "standard" (default), "minmax" or None
    numeric_impute="mean",  # "median" (default), "mean" or "most_frequent"
    drop_columns=["customer_id"],  # ignore ids and leakage columns
    return_dataframe=True,  # DataFrame with readable names, or NumPy array
)
prep.fit(X_train)
prep.transform(X_test)
prep.get_feature_names_out()  # ['age', 'income', 'city=Delhi', 'city=Pune', ...]
```

Column types are inferred from dtypes. Pass `numeric_columns=[...]` and `categorical_columns=[...]` to override. Categories unseen during `fit` become all-zero columns, so the output width never changes.

It works inside any scikit-learn pipeline:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(Preprocessor(), LogisticRegression())
model.fit(X_train, y_train)
model.score(X_test, y_test)
```

### MLPClassifier and MLPRegressor

```python
from mlkit.deep import MLPClassifier, MLPRegressor

clf = MLPClassifier(
    hidden_sizes=(128, 64),
    dropout=0.2,
    activation="relu",  # "relu", "tanh", "gelu" or "sigmoid"
    epochs=100,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-4,
    validation_fraction=0.2,  # hold out 20% for early stopping
    early_stopping_patience=5,  # stop after 5 epochs without improvement
    device="auto",  # "auto", "cpu" or "cuda"
    random_state=42,
    verbose=True,
)
clf.fit(X_train, y_train)  # labels can be strings or ints
clf.predict(X_test)  # original labels
clf.predict_proba(X_test)  # shape (n_samples, n_classes)
clf.history_["val_loss"]  # per-epoch losses

reg = MLPRegressor(hidden_sizes=(64,), epochs=80, random_state=0)
reg.fit(X_train, y_train)  # 1-d or 2-d targets
reg.predict(X_test)
```

Targets for `MLPRegressor` are standardised internally, so large offsets (for example prices in the thousands) train fine without manual scaling.

### CNNClassifier

```python
import numpy as np
from mlkit.deep import CNNClassifier

# images: (N, C, H, W) float array scaled to [0, 1]; (N, H, W) is treated as one channel
images = np.load("digits.npy") / 255.0
labels = np.load("labels.npy")

cnn = CNNClassifier(channels=(32, 64, 128), epochs=10, batch_size=128, random_state=0)
cnn.fit(images, labels)
cnn.score(images, labels)
```

Each block halves the image, so with three blocks the images must be at least 8 x 8. A global average pool means one trained model accepts any image size above that minimum.

### Saving and loading models

```python
clf.save("model.pt")
clf = MLPClassifier.load("model.pt", device="cpu")
```

Saved files hold only tensors and plain values and are loaded in PyTorch's safe `weights_only` mode. Only load files you created yourself.

### Building raw networks

If you want to write your own training loop, the network builders are exposed:

```python
from mlkit.deep import build_mlp, build_cnn

net = build_mlp(in_features=20, out_features=3, hidden_sizes=(64, 64), dropout=0.1)
cnn = build_cnn(in_channels=3, num_classes=10, channels=(32, 64))
```

---

## Project structure

```
mlkit-py/
├── mlkit/
│   ├── __init__.py        # public API
│   ├── preprocessing.py   # cleaning, outliers, splitting, Preprocessor
│   └── deep.py            # MLPClassifier, MLPRegressor, CNNClassifier
├── examples/
│   └── tabular_classification.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_deep.py
├── pyproject.toml
└── README.md
```

---

## Development

```bash
uv run pytest           # run the tests
uv run ruff check .     # lint
uv run ruff format .    # format
```

The deep learning tests are skipped automatically when PyTorch is not installed. CI runs lint and tests on every push. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

---

## Roadmap

- Classical model wrappers with sensible defaults (gradient boosting, random forests).
- Feature engineering helpers (date parts, target encoding, text vectorisation).
- Evaluation utilities: confusion matrix plots, regression diagnostics, cross-validation reports.
- Application templates for NLP and computer vision tasks.

This is a personal project. Issues and pull requests are read but not guaranteed a reply.

---

## License

MIT
