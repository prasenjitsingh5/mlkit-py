import numpy as np
import pandas as pd
import pytest

from aiml.preprocessing import (
    Preprocessor,
    clean_dataframe,
    remove_outliers,
    split_data,
    split_features_target,
)


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "age": [23, 45, None, 31, 31, 200],
            "income": [50.0, 80.0, 60.0, None, None, 70.0],
            "city": [" Delhi", "Pune", "Delhi", "", None, "Pune "],
            "bought": [0, 1, 0, 1, 1, 0],
        }
    )


# ----------------------------------------------------------------- cleaning
def test_clean_strips_and_blanks_to_na(frame):
    out = clean_dataframe(frame, drop_duplicates=False)
    assert out.loc[0, "city"] == "Delhi"
    assert out.loc[5, "city"] == "Pune"
    assert pd.isna(out.loc[3, "city"])
    assert pd.isna(out.loc[4, "city"])
    # never mutates the input
    assert frame.loc[0, "city"] == " Delhi"


def test_clean_drops_duplicates_and_na_rows():
    df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
    out = clean_dataframe(df)
    assert len(out) == 2
    out = clean_dataframe(df, drop_na_rows=True)
    assert len(out) == 1


def test_clean_drops_sparse_columns():
    df = pd.DataFrame({"good": [1, 2, 3, 4], "bad": [None, None, None, 1]})
    out = clean_dataframe(df, drop_na_columns_threshold=0.5)
    assert list(out.columns) == ["good"]


def test_clean_rejects_bad_threshold(frame):
    with pytest.raises(ValueError):
        clean_dataframe(frame, drop_na_columns_threshold=2)


# ----------------------------------------------------------------- outliers
def test_remove_outliers_iqr(frame):
    out = remove_outliers(frame, ["age"])
    assert 200 not in out["age"].values
    # NaN rows are kept for the imputer
    assert out["age"].isna().sum() == 1


def test_remove_outliers_zscore():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})
    out = remove_outliers(df, method="zscore", factor=2.0)
    assert 1000 not in out["x"].values
    assert len(out) == 5


def test_remove_outliers_bad_method(frame):
    with pytest.raises(ValueError):
        remove_outliers(frame, method="mad")


# ----------------------------------------------------------------- splitting
def test_split_features_target(frame):
    X, y = split_features_target(frame, "bought")
    assert "bought" not in X.columns
    assert y.name == "bought"
    with pytest.raises(KeyError):
        split_features_target(frame, "missing")


def test_split_data_stratified(frame):
    X_tr, X_te, y_tr, y_te = split_data(frame, "bought", test_size=0.5, stratify=True)
    assert len(X_tr) + len(X_te) == len(frame)
    assert set(y_tr.unique()) == {0, 1}
    assert set(y_te.unique()) == {0, 1}


# ----------------------------------------------------------------- Preprocessor
def test_preprocessor_infers_columns_and_names(frame):
    X, _ = split_features_target(clean_dataframe(frame, drop_duplicates=False), "bought")
    prep = Preprocessor().fit(X)
    assert prep.numeric_columns_ == ["age", "income"]
    assert prep.categorical_columns_ == ["city"]
    out = prep.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["age", "income", "city=Delhi", "city=Pune"]
    assert not out.isna().any().any()
    # standard scaling gives roughly zero mean
    assert abs(out["age"].mean()) < 1e-9


def test_preprocessor_handles_unknown_category(frame):
    X, _ = split_features_target(clean_dataframe(frame, drop_duplicates=False), "bought")
    prep = Preprocessor(scaler="minmax").fit(X)
    new = pd.DataFrame({"age": [40], "income": [65.0], "city": ["Mumbai"]})
    out = prep.transform(new)
    assert out.shape == (1, 4)
    assert out[["city=Delhi", "city=Pune"]].sum().sum() == 0
    assert 0 <= out.loc[0, "age"] <= 1


def test_preprocessor_drop_columns_and_array_output(frame):
    frame = frame.assign(id=range(len(frame)))
    X, _ = split_features_target(frame, "bought")
    prep = Preprocessor(drop_columns=["id"], scaler=None, return_dataframe=False).fit(X)
    out = prep.transform(X)
    assert isinstance(out, np.ndarray)
    assert "id" not in prep.get_feature_names_out()


def test_preprocessor_errors():
    prep = Preprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(pd.DataFrame({"a": [1]}))
    with pytest.raises(TypeError):
        prep.fit(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        Preprocessor(scaler="log").fit(pd.DataFrame({"a": [1.0, 2.0]}))
    fitted = Preprocessor().fit(pd.DataFrame({"a": [1.0, 2.0]}))
    with pytest.raises(KeyError):
        fitted.transform(pd.DataFrame({"b": [1.0]}))


def test_preprocessor_works_in_sklearn_pipeline(frame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    df = clean_dataframe(frame, drop_duplicates=False)
    X, y = split_features_target(df, "bought")
    model = make_pipeline(Preprocessor(), LogisticRegression())
    model.fit(X, y)
    assert model.predict(X).shape == (len(X),)
