"""Deep learning models with a scikit-learn style ``fit`` / ``predict`` API.

Three estimators are provided, all built on PyTorch:

* :class:`MLPClassifier` for tabular classification (binary or multi-class).
* :class:`MLPRegressor` for tabular regression (single or multi-output).
* :class:`CNNClassifier` for image classification on ``(N, C, H, W)`` arrays.

PyTorch is an optional dependency. Install it with ``pip install mlkit-py[deep]``.
Everything in this module accepts NumPy arrays, pandas objects or torch
tensors, trains on CPU by default and picks a GPU automatically when one is
available (``device="auto"``).

Example
-------
>>> import numpy as np
>>> from mlkit.deep import MLPClassifier
>>> rng = np.random.default_rng(0)
>>> X = rng.normal(size=(200, 4))
>>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
>>> clf = MLPClassifier(hidden_sizes=(16,), epochs=30, random_state=0).fit(X, y)
>>> clf.score(X, y) > 0.8
True
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = ["MLPClassifier", "MLPRegressor", "CNNClassifier", "build_mlp", "build_cnn"]


# --------------------------------------------------------------------------- #
# Optional torch import
# --------------------------------------------------------------------------- #
def _torch():
    try:
        import torch  # noqa: WPS433
        import torch.nn as nn  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "PyTorch is required for mlkit.deep. Install it with `pip install mlkit-py[deep]` "
            "or follow https://pytorch.org/get-started/locally/."
        ) from exc
    return torch, nn


def _resolve_device(device: str):
    torch, _ = _torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_numpy(a) -> np.ndarray:
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    if hasattr(a, "to_numpy"):
        return a.to_numpy()
    return np.asarray(a)


# --------------------------------------------------------------------------- #
# Network builders
# --------------------------------------------------------------------------- #
def build_mlp(
    in_features: int,
    out_features: int,
    hidden_sizes: Sequence[int] = (64, 64),
    *,
    dropout: float = 0.0,
    activation: str = "relu",
):
    """Return a fully connected ``torch.nn.Sequential`` network."""
    _, nn = _torch()
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid}
    if activation not in acts:
        raise ValueError(f"activation must be one of {sorted(acts)}")
    layers = []
    prev = in_features
    for size in hidden_sizes:
        layers.append(nn.Linear(prev, size))
        layers.append(acts[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = size
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


def build_cnn(
    in_channels: int,
    num_classes: int,
    *,
    channels: Sequence[int] = (32, 64),
    kernel_size: int = 3,
    dropout: float = 0.0,
):
    """Return a small convolutional network.

    Each block is ``Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2)``. A global
    average pool makes the head independent of input height and width.
    """
    _, nn = _torch()
    blocks = []
    prev = in_channels
    for ch in channels:
        blocks.extend(
            [
                nn.Conv2d(prev, ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        )
        prev = ch
    head = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    if dropout > 0:
        head.append(nn.Dropout(dropout))
    head.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*blocks, *head)


# --------------------------------------------------------------------------- #
# Shared training loop
# --------------------------------------------------------------------------- #
class _TorchEstimator(BaseEstimator):
    """Common fit / predict machinery. Subclasses implement the task specifics."""

    def __init__(
        self,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        validation_fraction: float = 0.0,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ---- hooks for subclasses -------------------------------------------
    def _prepare_X(self, X, fitting: bool):
        raise NotImplementedError

    def _prepare_y(self, y, fitting: bool):
        raise NotImplementedError

    def _build_network(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    # ---- training --------------------------------------------------------
    def fit(self, X, y):
        torch, _ = _torch()
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")

        self.device_ = _resolve_device(self.device)
        X_t = self._prepare_X(X, fitting=True)
        y_t = self._prepare_y(y, fitting=True)
        if len(X_t) != len(y_t):
            raise ValueError("X and y have different lengths")

        # Optional hold-out split for early stopping.
        n = len(X_t)
        n_val = int(round(n * self.validation_fraction))
        perm = torch.randperm(n)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        if len(train_idx) == 0:
            raise ValueError("no training samples left after validation split")

        self.network_ = self._build_network().to(self.device_)
        loss_fn = self._loss()
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_state, best_val, bad_epochs = None, float("inf"), 0
        self.history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.network_.train()
            order = train_idx[torch.randperm(len(train_idx))]
            total, seen = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = X_t[idx].to(self.device_)
                yb = y_t[idx].to(self.device_)
                optimizer.zero_grad()
                loss = loss_fn(self.network_(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
                seen += len(idx)
            train_loss = total / seen
            self.history_["train_loss"].append(train_loss)

            val_loss = None
            if n_val:
                val_loss = self._evaluate_loss(X_t[val_idx], y_t[val_idx], loss_fn)
                self.history_["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-8:
                    best_val, bad_epochs = val_loss, 0
                    best_state = copy.deepcopy(self.network_.state_dict())
                else:
                    bad_epochs += 1

            if self.verbose:
                msg = f"epoch {epoch + 1}/{self.epochs} train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" val_loss={val_loss:.4f}"
                print(msg)

            if (
                self.early_stopping_patience is not None
                and n_val
                and bad_epochs >= self.early_stopping_patience
            ):
                break

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.n_epochs_ = len(self.history_["train_loss"])
        return self

    def _evaluate_loss(self, X_t, y_t, loss_fn) -> float:
        torch, _ = _torch()
        self.network_.eval()
        total, seen = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                xb = X_t[start : start + self.batch_size].to(self.device_)
                yb = y_t[start : start + self.batch_size].to(self.device_)
                total += loss_fn(self.network_(xb), yb).item() * len(xb)
                seen += len(xb)
        return total / seen

    def _check_fitted(self) -> None:
        if not hasattr(self, "network_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted; call fit first")

    def _forward(self, X) -> np.ndarray:
        torch, _ = _torch()
        self._check_fitted()
        X_t = self._prepare_X(X, fitting=False)
        self.network_.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(X_t), self.batch_size):
                xb = X_t[start : start + self.batch_size].to(self.device_)
                outs.append(self.network_(xb).cpu())
        return torch.cat(outs).numpy()

    # ---- persistence -----------------------------------------------------
    @staticmethod
    def _pack(value):
        """Convert fitted attributes to plain Python so files load with ``weights_only=True``."""
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: _TorchEstimator._pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_TorchEstimator._pack(v) for v in value]
        return value

    @staticmethod
    def _unpack(value):
        if isinstance(value, dict):
            if "__ndarray__" in value:
                return np.asarray(value["__ndarray__"], dtype=value["dtype"])
            return {k: _TorchEstimator._unpack(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_TorchEstimator._unpack(v) for v in value]
        return value

    def save(self, path: str) -> None:
        """Save hyper-parameters, fitted attributes and weights to ``path``.

        The file contains only tensors and plain Python values, so it can be
        loaded with ``torch.load(..., weights_only=True)`` and never executes
        arbitrary code.
        """
        torch, _ = _torch()
        if not hasattr(self, "network_"):
            raise RuntimeError("cannot save an unfitted model")
        fitted = {
            k: self._pack(v)
            for k, v in vars(self).items()
            if k.endswith("_") and k not in {"network_", "device_"}
        }
        torch.save(
            {
                "class": type(self).__name__,
                "params": self._pack(self.get_params()),
                "fitted": fitted,
                "state_dict": self.network_.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, *, device: str = "auto"):
        """Load a model previously written by :meth:`save`.

        Only trust files you created yourself. Loading uses the safe
        ``weights_only`` mode, so a tampered file fails instead of running code.
        """
        torch, _ = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("class") != cls.__name__:
            raise TypeError(f"file holds a {payload.get('class')}, not a {cls.__name__}")
        params = cls._unpack(payload["params"])
        for key in ("hidden_sizes", "channels"):
            if key in params and isinstance(params[key], list):
                params[key] = tuple(params[key])
        model = cls(**params)
        model.device = device
        for k, v in payload["fitted"].items():
            setattr(model, k, cls._unpack(v))
        model.device_ = _resolve_device(device)
        model.network_ = model._build_network().to(model.device_)
        model.network_.load_state_dict(payload["state_dict"])
        model.network_.eval()
        return model


# --------------------------------------------------------------------------- #
# Tabular helpers
# --------------------------------------------------------------------------- #
class _TabularMixin:
    def _prepare_X(self, X, fitting: bool):
        torch, _ = _torch()
        arr = _to_numpy(X).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features)")
        if fitting:
            self.n_features_in_ = arr.shape[1]
        elif arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {arr.shape[1]} features, expected {self.n_features_in_}")
        return torch.from_numpy(np.ascontiguousarray(arr))


class MLPClassifier(_TabularMixin, ClassifierMixin, _TorchEstimator):
    """Multi-layer perceptron classifier for tabular data.

    Labels can be any hashable values; they are mapped to integer indices
    internally and returned unchanged by :meth:`predict`.

    Parameters
    ----------
    hidden_sizes:
        Width of each hidden layer.
    dropout:
        Dropout probability applied after every hidden layer.
    activation:
        ``"relu"``, ``"tanh"``, ``"gelu"`` or ``"sigmoid"``.
    epochs, batch_size, lr, weight_decay:
        Standard Adam training settings.
    validation_fraction, early_stopping_patience:
        Hold out a fraction of the training set and stop when its loss has
        not improved for ``early_stopping_patience`` epochs. The best weights
        are restored at the end.
    device:
        ``"auto"``, ``"cpu"`` or ``"cuda"``.
    random_state:
        Seed for reproducible initialisation and shuffling.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (64, 64),
        *,
        dropout: float = 0.0,
        activation: str = "relu",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        validation_fraction: float = 0.0,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.activation = activation

    def _prepare_y(self, y, fitting: bool):
        torch, _ = _torch()
        arr = _to_numpy(y).ravel()
        if fitting:
            self.classes_ = np.unique(arr)
            if len(self.classes_) < 2:
                raise ValueError("need at least two classes")
        lookup = {c: i for i, c in enumerate(self.classes_)}
        try:
            idx = np.array([lookup[v] for v in arr], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"unknown label {exc.args[0]!r}") from exc
        return torch.from_numpy(idx)

    def _build_network(self):
        return build_mlp(
            self.n_features_in_,
            len(self.classes_),
            self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
        )

    def _loss(self):
        _, nn = _torch()
        return nn.CrossEntropyLoss()

    def predict_proba(self, X) -> np.ndarray:
        torch, _ = _torch()
        self._check_fitted()
        logits = torch.from_numpy(self._forward(X))
        return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class MLPRegressor(_TabularMixin, RegressorMixin, _TorchEstimator):
    """Multi-layer perceptron regressor for tabular data.

    Supports a single target (1-d ``y``) or several targets (2-d ``y``).
    Targets are standardised internally so the learning rate behaves the same
    regardless of their scale. See :class:`MLPClassifier` for the parameters.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (64, 64),
        *,
        dropout: float = 0.0,
        activation: str = "relu",
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        validation_fraction: float = 0.0,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.activation = activation

    def _prepare_y(self, y, fitting: bool):
        torch, _ = _torch()
        arr = _to_numpy(y).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if fitting:
            self.n_outputs_ = arr.shape[1]
            self.y_mean_ = arr.mean(axis=0)
            self.y_std_ = arr.std(axis=0) + 1e-8
        arr = (arr - self.y_mean_) / self.y_std_
        return torch.from_numpy(np.ascontiguousarray(arr))

    def _build_network(self):
        return build_mlp(
            self.n_features_in_,
            self.n_outputs_,
            self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
        )

    def _loss(self):
        _, nn = _torch()
        return nn.MSELoss()

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        out = self._forward(X) * self.y_std_ + self.y_mean_
        return out.ravel() if self.n_outputs_ == 1 else out


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
class CNNClassifier(ClassifierMixin, _TorchEstimator):
    """Convolutional classifier for image arrays shaped ``(N, C, H, W)``.

    A 3-d array ``(N, H, W)`` is treated as single-channel. Pixel values are
    used as given; scale them to ``[0, 1]`` beforehand for best results.

    Parameters
    ----------
    channels:
        Output channels of each convolutional block. Every block halves the
        spatial size, so ``H`` and ``W`` must be at least ``2 ** len(channels)``.
    kernel_size:
        Convolution kernel size (odd numbers keep the spatial size).
    dropout:
        Dropout before the final linear layer.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64),
        *,
        kernel_size: int = 3,
        dropout: float = 0.0,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        validation_fraction: float = 0.0,
        early_stopping_patience: int | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout

    def _prepare_X(self, X, fitting: bool):
        torch, _ = _torch()
        arr = _to_numpy(X).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[:, None, :, :]
        if arr.ndim != 4:
            raise ValueError("X must be shaped (N, C, H, W) or (N, H, W)")
        min_side = 2 ** len(self.channels)
        if arr.shape[2] < min_side or arr.shape[3] < min_side:
            raise ValueError(f"images must be at least {min_side}x{min_side} for {len(self.channels)} blocks")
        if fitting:
            self.in_channels_ = arr.shape[1]
        elif arr.shape[1] != self.in_channels_:
            raise ValueError(f"X has {arr.shape[1]} channels, expected {self.in_channels_}")
        return torch.from_numpy(np.ascontiguousarray(arr))

    _prepare_y = MLPClassifier._prepare_y

    def _build_network(self):
        return build_cnn(
            self.in_channels_,
            len(self.classes_),
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

    def _loss(self):
        _, nn = _torch()
        return nn.CrossEntropyLoss()

    predict_proba = MLPClassifier.predict_proba
    predict = MLPClassifier.predict
