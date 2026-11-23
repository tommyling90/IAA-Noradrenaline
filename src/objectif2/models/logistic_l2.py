"""Régression logistique L2 — référence linéaire probabiliste"""
from sklearn.linear_model import LogisticRegression


def get_model(C=1.0, class_weight=None):
    return LogisticRegression(
        C=C, class_weight=class_weight,
        solver="lbfgs", max_iter=1000
    )