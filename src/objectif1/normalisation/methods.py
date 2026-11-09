"""Add your own normalizer here. Normalizer should not be the same across all ML methods
"""
from sklearn.preprocessing import StandardScaler, FunctionTransformer


def get_normalizer(name):
    if name == "standard":
        return StandardScaler()
    if name == "none": #default en ce moment
        return FunctionTransformer(validate=False)
    raise ValueError(f"Unknown normalization method: {name}")