"""
Single-row RandomForest/ExtraTrees regression without sklearn's per-tree
overhead.

Frontier predicts each batch's time with forests of 250-750 trees, one row at a
time. sklearn spends ~40us of Python overhead per tree on a single row (~10ms a
prediction), which made up ~90% of a Frontier episode. This walks every tree at
once with numpy and sums the leaves in sklearn's order, so the result is
bit-identical (~0.1-0.2ms a prediction). Anything unusual (several rows, sparse,
non-finite, mismatched features, multi-output, unfitted) goes to sklearn, so it
returns or raises what sklearn would.

Bit-identity relies on sklearn's tree layout and its float32 cast of inputs
(checked on sklearn 1.9.1; tests/test_fast_forest.py).
"""

import numpy as np
import scipy.sparse as sp
import sklearn.ensemble._forest as F

_sklearn_predict = F.ForestRegressor.predict


def _compile(forest):
    """All trees' nodes in flat arrays; children index the flat arrays, -1 at leaves."""
    trees = [e.tree_ for e in forest.estimators_]
    sizes = np.array([t.node_count for t in trees])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    left = np.concatenate(
        [np.where(t.children_left < 0, -1, t.children_left + o) for t, o in zip(trees, offsets)]
    )
    right = np.concatenate(
        [np.where(t.children_right < 0, -1, t.children_right + o) for t, o in zip(trees, offsets)]
    )
    feature = np.concatenate([np.maximum(t.feature, 0) for t in trees])
    threshold = np.concatenate([t.threshold for t in trees])
    value = np.concatenate([t.value[:, 0, 0] for t in trees])
    # refits replace estimators_; warm starts append to it
    key = (forest.estimators_, len(forest.estimators_))
    return key, (offsets, left, right, feature, threshold, value)


def _row(forest, X):
    """X as one row of float32 values (held in float64) if the fast path applies, else None."""
    if not hasattr(forest, "estimators_") or forest.n_outputs_ != 1 or sp.issparse(X):
        return None
    names = getattr(forest, "feature_names_in_", None)
    columns = getattr(X, "columns", None)
    if names is not None and columns is not None and list(columns) != list(names):
        return None
    try:
        x = np.asarray(X)
    except Exception:
        return None
    if x.ndim != 2 or x.shape != (1, forest.n_features_in_) or x.dtype.kind not in "biuf":
        return None
    with np.errstate(over="ignore"):
        x = x[0].astype(np.float32)  # sklearn compares float32 inputs
    if not np.isfinite(x).all():
        return None
    return x.astype(np.float64)


def predict(forest, X):
    x = _row(forest, X)
    if x is None:
        return _sklearn_predict(forest, X)
    compiled = forest.__dict__.get("_fast_forest")
    if (
        compiled is None
        or compiled[0][0] is not forest.estimators_
        or compiled[0][1] != len(forest.estimators_)
    ):
        compiled = forest.__dict__["_fast_forest"] = _compile(forest)
    roots, left, right, feature, threshold, value = compiled[1]
    nodes = roots
    while True:
        children = left[nodes]
        inner = children >= 0
        if not inner.any():
            break
        step = np.where(x[feature[nodes]] <= threshold[nodes], children, right[nodes])
        nodes = np.where(inner, step, nodes)
    total = 0.0
    for v in value[nodes]:  # sklearn's summation order
        total += v
    return np.array([total / len(roots)])


def install() -> None:
    F.ForestRegressor.predict = predict


def uninstall() -> None:
    F.ForestRegressor.predict = _sklearn_predict
