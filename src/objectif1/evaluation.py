import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from .normalisation.methods import get_normalizer

def blocked_splits(centers, n_splits=5, radius=3):
    centers = np.asarray(centers)
    if len(centers) < n_splits:
        raise ValueError("Too few frames for CV")

    # create 5 blocks from "centers"
    blocks = np.array_split(np.arange(len(centers)), n_splits)

    for block in blocks:
        lo, hi = centers[block].min(), centers[block].max()
        # pas de chevauchement entre train et trest
        # train contains the indices of centers and not centers values themselves
        # block also contains the indices of centers. dim(block) = dim(centers) since centers is returned after [valid] mask
        train = np.where((centers < lo - 2*radius) | (centers > hi + 2*radius))[0]
        if len(train) == 0:
            raise ValueError("No training rows left after temporal purging")
        yield train, block


def evaluate_model(X, y, centers, estimator, grid, normalisation="standard", outer_folds=5, inner_folds=3, radius=3):
    pred = np.full(len(y), np.nan)
    best_params = []
    for train, test in blocked_splits(centers, outer_folds, radius):
        inner = list(blocked_splits(centers[train], inner_folds, radius))
        model = Pipeline([
            ("normalizer", get_normalizer(normalisation)),
            ("regressor", estimator)
        ])
        search = GridSearchCV(model, grid, cv=inner, scoring="r2", n_jobs=-1,
                              refit=True, error_score="raise")
        search.fit(X[train], y[train])
        pred[test] = search.predict(X[test])
        best_params.append(search.best_params_)
    valid = np.isfinite(pred)
    if valid.sum() < 2:
        raise ValueError("Insufficient valid predictions")

    corr = float(np.corrcoef(y[valid], pred[valid])[0,1]) if np.std(pred[valid]) > 0 and np.std(y[valid]) > 0 else float("nan")
    return {
        "r2": float(r2_score(y[valid], pred[valid])),
        "correlation": corr,
        "best_params_per_test_block": best_params,
        "predictions": pred
    }