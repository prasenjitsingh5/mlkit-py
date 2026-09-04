import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aiml.deep import CNNClassifier, MLPClassifier, MLPRegressor, build_cnn, build_mlp


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    y = np.where(X[:, 0] + X[:, 1] > 0, "yes", "no")
    return X, y


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 100
    return X, y


# ----------------------------------------------------------------- builders
def test_build_mlp_shapes():
    net = build_mlp(4, 3, (8, 8), dropout=0.1)
    out = net(torch.zeros(2, 4))
    assert out.shape == (2, 3)
    with pytest.raises(ValueError):
        build_mlp(4, 3, activation="swish")


def test_build_cnn_is_size_agnostic():
    net = build_cnn(1, 2, channels=(4, 8))
    assert net(torch.zeros(2, 1, 8, 8)).shape == (2, 2)
    assert net(torch.zeros(2, 1, 12, 16)).shape == (2, 2)


# ----------------------------------------------------------------- MLPClassifier
def test_mlp_classifier_learns_and_keeps_labels(clf_data):
    X, y = clf_data
    clf = MLPClassifier(hidden_sizes=(16,), epochs=40, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.85
    assert set(clf.predict(X)) <= {"yes", "no"}
    proba = clf.predict_proba(X[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_mlp_classifier_accepts_dataframe_and_reproduces(clf_data):
    X, y = clf_data
    df = pd.DataFrame(X, columns=list("abcde"))
    a = MLPClassifier(hidden_sizes=(8,), epochs=5, random_state=7).fit(df, pd.Series(y))
    b = MLPClassifier(hidden_sizes=(8,), epochs=5, random_state=7).fit(df, pd.Series(y))
    assert np.array_equal(a.predict(df), b.predict(df))


def test_mlp_classifier_early_stopping(clf_data):
    X, y = clf_data
    # Flip a fifth of the labels so validation loss bottoms out early.
    rng = np.random.default_rng(3)
    y = y.copy()
    flip = rng.random(len(y)) < 0.2
    y[flip] = np.where(y[flip] == "yes", "no", "yes")
    clf = MLPClassifier(
        hidden_sizes=(32, 32),
        epochs=200,
        lr=1e-2,
        validation_fraction=0.3,
        early_stopping_patience=3,
        random_state=0,
    ).fit(X, y)
    assert clf.n_epochs_ < 200
    assert len(clf.history_["val_loss"]) == clf.n_epochs_


def test_mlp_classifier_errors(clf_data):
    X, y = clf_data
    with pytest.raises(RuntimeError):
        MLPClassifier().predict(X)
    with pytest.raises(ValueError):
        MLPClassifier(epochs=1).fit(X, np.zeros(len(X)))  # one class
    with pytest.raises(ValueError):
        MLPClassifier(epochs=1).fit(X, y[:10])  # length mismatch
    clf = MLPClassifier(epochs=1, random_state=0).fit(X, y)
    with pytest.raises(ValueError):
        clf.predict(X[:, :3])  # wrong feature count


def test_mlp_classifier_save_and_load(tmp_path, clf_data):
    X, y = clf_data
    clf = MLPClassifier(hidden_sizes=(8,), epochs=5, random_state=0).fit(X, y)
    path = tmp_path / "clf.pt"
    clf.save(str(path))
    loaded = MLPClassifier.load(str(path), device="cpu")
    assert np.array_equal(loaded.predict(X), clf.predict(X))
    assert loaded.hidden_sizes == (8,)
    assert list(loaded.classes_) == ["no", "yes"]
    with pytest.raises(TypeError):
        MLPRegressor.load(str(path))


def test_saved_file_loads_in_safe_mode(tmp_path, reg_data):
    X, y = reg_data
    reg = MLPRegressor(hidden_sizes=(4,), epochs=2, random_state=0).fit(X, y)
    path = tmp_path / "reg.pt"
    reg.save(str(path))
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    assert payload["class"] == "MLPRegressor"
    loaded = MLPRegressor.load(str(path), device="cpu")
    assert np.allclose(loaded.predict(X), reg.predict(X))


def test_cnn_save_and_load(tmp_path):
    X = np.zeros((8, 1, 8, 8), dtype=np.float32)
    X[4:, :, 2:6, 2:6] = 1.0
    y = [0] * 4 + [1] * 4
    cnn = CNNClassifier(channels=(4,), epochs=2, batch_size=4, random_state=0).fit(X, y)
    path = tmp_path / "cnn.pt"
    cnn.save(str(path))
    loaded = CNNClassifier.load(str(path), device="cpu")
    assert np.array_equal(loaded.predict(X), cnn.predict(X))


# ----------------------------------------------------------------- MLPRegressor
def test_mlp_regressor_learns_offset_targets(reg_data):
    X, y = reg_data
    reg = MLPRegressor(hidden_sizes=(32,), epochs=60, random_state=0).fit(X, y)
    assert reg.score(X, y) > 0.9
    assert reg.predict(X).shape == (len(X),)


def test_mlp_regressor_multi_output(reg_data):
    X, y = reg_data
    Y = np.column_stack([y, -y])
    reg = MLPRegressor(hidden_sizes=(16,), epochs=10, random_state=0).fit(X, Y)
    assert reg.predict(X).shape == (len(X), 2)


# ----------------------------------------------------------------- CNNClassifier
def test_cnn_classifier_on_synthetic_images():
    rng = np.random.default_rng(0)
    n = 120
    imgs = rng.normal(scale=0.1, size=(n, 8, 8)).astype(np.float32)
    labels = rng.integers(0, 2, size=n)
    imgs[labels == 1, 2:6, 2:6] += 1.0  # bright square marks class 1
    clf = CNNClassifier(channels=(4, 8), epochs=15, batch_size=16, random_state=0).fit(imgs, labels)
    assert clf.score(imgs, labels) > 0.9
    assert clf.predict_proba(imgs[:3]).shape == (3, 2)


def test_cnn_classifier_rejects_tiny_images():
    X = np.zeros((4, 1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        CNNClassifier(channels=(4, 8), epochs=1).fit(X, [0, 1, 0, 1])
