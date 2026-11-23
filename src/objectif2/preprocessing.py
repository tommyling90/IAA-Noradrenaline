"""
Objectif 2 — preprocessing.py
Sous-échantillonnage des neurones et mise en forme des features.
"""
import numpy as np


def sample_neurons(X, n_samples, rng):
    """
    Sous-échantillonne les neurones aléatoirement.
    X : (n_trials, n_neurons, n_frames)
    Retourne : (n_trials, n_samples, n_frames), indices
    """
    n_neurons = X.shape[1]
    if n_samples > n_neurons:
        raise ValueError(f"n_samples={n_samples} > n_neurons={n_neurons}")
    indices = rng.choice(n_neurons, size=n_samples, replace=False)
    return X[:, indices, :], indices


def flatten_trials(X):
    """
    Aplatit (n_trials, n_neurons, n_frames) -> (n_trials, n_neurons * n_frames)
    pour les modèles qui attendent une entrée 2D.
    """
    n_trials = X.shape[0]
    return X.reshape(n_trials, -1)


def standardize(X_train, X_test):
    """
    Standardisation (x - mean) / std calculée sur X_train uniquement.
    Appliquée APRES le fold pour éviter la fuite de données.
    X : (n_trials, n_features)
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std == 0] = 1.0  # éviter division par zéro
    return (X_train - mean) / std, (X_test - mean) / std


def compute_class_weights(y):
    """
    Calcule les poids de classe inversement proportionnels à leur fréquence.
    Retourne un dict {0: w0, 1: w1} pour sklearn.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(c): total / (len(classes) * cnt)
               for c, cnt in zip(classes, counts)}
    return weights