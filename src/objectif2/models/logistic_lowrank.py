"""Régression logistique de faible rang."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted


class LowRankLogistic(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=10, C=1.0, class_weight=None):
        self.n_components = n_components
        self.C = C
        self.class_weight = class_weight

    def fit(self, X, y):
        self.svd_ = TruncatedSVD(n_components=self.n_components)
        X_reduced = self.svd_.fit_transform(X)
        self.lr_ = LogisticRegression(
            C=self.C, class_weight=self.class_weight,
            solver="lbfgs", max_iter=1000
        )
        self.lr_.fit(X_reduced, y)
        self.classes_ = self.lr_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.lr_.predict(self.svd_.transform(X))

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.lr_.predict_proba(self.svd_.transform(X))