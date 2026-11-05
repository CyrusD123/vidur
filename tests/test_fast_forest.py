"""
vidur/rl/fast_forest.py must match sklearn's ForestRegressor.predict bit for
bit, or hand the call to sklearn (same result or same exception).

    python -m pytest tests/test_fast_forest.py -v
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from vidur.rl import fast_forest

rng = np.random.default_rng(0)
X = np.vstack([rng.lognormal(3, 2, size=(1500, 6)), rng.integers(0, 50, size=(300, 6))]).astype(float)
X[:, 3] = np.round(X[:, 3])  # many ties
X[:, 4] = 1e6 + rng.normal(size=len(X)) * 1e-3  # finer than float32 resolves
Y = np.log1p(X[:, 0]) * X[:, 1] ** 0.5 + rng.normal(size=len(X))
NAMES = list("abcdef")


def _outcome(fn, forest, x):
    try:
        return fn(forest, x)
    except Exception as exc:
        return type(exc)


def _assert_same(forest, x):
    expected = _outcome(fast_forest._sklearn_predict, forest, x)
    got = _outcome(fast_forest.predict, forest, x)
    if isinstance(expected, type):
        assert got is expected
    else:
        assert isinstance(got, np.ndarray)
        assert got.shape == expected.shape and got.dtype == expected.dtype
        assert np.array_equal(got, expected)


def _test_rows(forest, n_boundary=500, n_random=200):
    """Training rows, rows with one feature on or next to a split threshold, random and extreme rows."""
    rows = list(X[:100])
    trees = [e.tree_ for e in forest.estimators_]
    for _ in range(n_boundary):
        tree = trees[rng.integers(len(trees))]
        inner = np.flatnonzero(tree.children_left >= 0)
        if not len(inner):
            break
        node = rng.choice(inner)
        th, th32 = tree.threshold[node], np.float32(tree.threshold[node])
        row = X[rng.integers(len(X))].copy()
        row[tree.feature[node]] = rng.choice(
            [th, float(th32), float(np.nextafter(th32, np.float32(np.inf))),
             float(np.nextafter(th32, np.float32(-np.inf))), np.nextafter(th, np.inf), np.nextafter(th, -np.inf)]
        )
        rows.append(row)
    rows += list(rng.uniform(X.min(0) - 1, X.max(0) * 2, size=(n_random, X.shape[1])))
    rows += [np.full(X.shape[1], v) for v in (0.0, -1.0, 1e-40, 3e38, -3e38)]
    return rows


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (RandomForestRegressor, dict(n_estimators=30)),
        (RandomForestRegressor, dict(n_estimators=20, max_depth=4)),
        (RandomForestRegressor, dict(n_estimators=20, min_samples_leaf=7, max_features=0.5)),
        (RandomForestRegressor, dict(n_estimators=10, bootstrap=False)),
        (RandomForestRegressor, dict(n_estimators=1)),
        (RandomForestRegressor, dict(n_estimators=10, max_depth=1)),
        (RandomForestRegressor, dict(n_estimators=10, criterion="absolute_error", max_depth=6)),
        (ExtraTreesRegressor, dict(n_estimators=30)),
    ],
)
@pytest.mark.parametrize("with_names", [False, True])
def test_bit_identical(cls, kwargs, with_names):
    forest = cls(random_state=0, **kwargs).fit(pd.DataFrame(X, columns=NAMES) if with_names else X, Y)
    for row in _test_rows(forest):
        _assert_same(forest, pd.DataFrame([row], columns=NAMES) if with_names else row[None, :])


def test_single_leaf_trees():
    forest = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, np.ones(len(X)))
    _assert_same(forest, X[:1])


def test_trained_with_missing_values():
    Xn = X.copy()
    Xn[rng.random(Xn.shape) < 0.1] = np.nan
    forest = RandomForestRegressor(n_estimators=10, random_state=0).fit(Xn, Y)
    for i in range(200):
        _assert_same(forest, Xn[i : i + 1])  # NaN rows go to sklearn


def test_refit_recompiles():
    forest = RandomForestRegressor(n_estimators=5, warm_start=True, random_state=0).fit(X, Y)
    fast_forest.predict(forest, X[:1])
    forest.set_params(n_estimators=12).fit(X, Y)
    _assert_same(forest, X[:1])
    forest.set_params(warm_start=False).fit(X, -Y)
    _assert_same(forest, X[:1])


FOREST = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, Y)
NAMED = RandomForestRegressor(n_estimators=10, random_state=0).fit(pd.DataFrame(X, columns=NAMES), Y)
MULTI = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, np.c_[Y, -Y])


def _with(value):
    row = X[:1].copy()
    row[0, 2] = value
    return row


@pytest.mark.parametrize(
    "forest,x",
    [
        (FOREST, X[:2]),
        (FOREST, X[:0]),
        (FOREST, X[0]),
        (FOREST, X[:1, :5]),
        (FOREST, np.c_[X[:1], [[1.0]]]),
        (FOREST, X[:1].tolist()),
        (FOREST, X[:1].astype(int)),
        (FOREST, X[:1].astype(np.float32)),
        (FOREST, X[:1].astype(object)),
        (FOREST, np.array([["1"] * 6])),
        (FOREST, _with(np.nan)),
        (FOREST, _with(np.inf)),
        (FOREST, _with(1e39)),
        (FOREST, sp.csr_matrix(X[:1])),
        (MULTI, X[:1]),
        (NAMED, pd.DataFrame(X[:1], columns=NAMES)),
        (NAMED, pd.DataFrame(X[:1], columns=NAMES[::-1])),
        (NAMED, pd.DataFrame(X[:1], columns=NAMES[:-1] + ["Z"])),
        (NAMED, X[:1]),
        (FOREST, pd.DataFrame(X[:1], columns=NAMES)),
        (RandomForestRegressor(), X[:1]),
    ],
    ids=[
        "two-rows", "zero-rows", "1d", "too-few-features", "too-many-features", "list", "int",
        "float32", "object", "string", "nan", "inf", "float32-overflow", "sparse", "multi-output",
        "named", "named-reordered", "named-renamed", "named-ndarray", "unnamed-dataframe", "unfitted",
    ],
)
def test_edge_cases_match_sklearn(forest, x):
    _assert_same(forest, x)


def test_install_uninstall():
    from sklearn.ensemble._forest import ForestRegressor

    original = ForestRegressor.predict
    try:
        fast_forest.install()
        assert ForestRegressor.predict is fast_forest.predict
        fast_forest.uninstall()
        assert ForestRegressor.predict is fast_forest._sklearn_predict
    finally:
        ForestRegressor.predict = original
