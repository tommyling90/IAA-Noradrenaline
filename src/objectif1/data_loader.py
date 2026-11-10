"""Load extracted data from .npz file and find minimum number of neurons from all relevant Obj1 data"""
from pathlib import Path
import numpy as np

"""pour lister tous les paths comme arg pour minimum_neurons"""
def list_all_paths(preprocessed_dir):
    return sorted(Path(preprocessed_dir).glob("*.npz"))

def load_session(path):
    with np.load(path, allow_pickle=False) as obj:
        fluo = obj["fluo"].copy()
        lever_2p = obj["lever_2p"].copy()
    if fluo.ndim != 2 or lever_2p.ndim != 2:
        raise ValueError(f"Expected (neurons, frames) and (positions, frames): {path}")
    if fluo.shape[1] != lever_2p.shape[1]:
        raise ValueError(f"Time mismatch: {path}: {fluo.shape}, {lever_2p.shape}")
    return fluo, lever_2p

def minimum_neurons(paths):
    if not paths:
        raise ValueError("No extracted sessions found; run extraction notebook first")
    return min(load_session(p)[0].shape[0] for p in paths)