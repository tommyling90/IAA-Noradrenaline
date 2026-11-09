import warnings
import numpy as np

def average_lever(lever_2p):
    a = np.asarray(lever_2p, dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=0)

def sample_neurons(fluo, n_samples, rng):
    indices = rng.choice(fluo.shape[0], size=n_samples, replace=False)
    return fluo[indices], indices

def temporal_features(fluo, lever, radius=3):
