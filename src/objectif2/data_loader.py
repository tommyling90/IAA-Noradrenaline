"""
Objectif 2 — data_loader.py
Charge les sessions brutes et extrait les essais (hit/miss) depuis trial_matrix.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Sessions cibles : (groupe, souris, spot, [jours])
SESSIONS_CONFIG = [
    ("ctrl",  "1384", "Spot 1", [1, 13]),
    ("1a ko", "1382", "Spot 1", [1, 13]),
    ("1a ko", "1382", "Spot 2", [2, 12]),
    ("1a ko", "1386", "Spot 1", [1, 13]),
    ("1a ko", "1386", "Spot 2", [2, 14]),
    ("1a ko", "1385", "Spot 1", [1, 13]),
    ("1a ko", "1385", "Spot 2", [2, 14]),
]

BUFFER_FRAMES = 500  # frames tampon à retirer en fin de session


def load_session(raw_dir: Path, group: str, mouse: str,
                 spot: str, day: int) -> dict:
    """
    Charge fluo + trial_matrix pour une session donnée.
    Retourne un dict avec toutes les infos nécessaires à l'objectif 2.
    """
    spot_dir = raw_dir / group / mouse / spot
    path_2ph = spot_dir / f"D{day}_2PHData.npy"

    if not path_2ph.exists():
        raise FileNotFoundError(f"Fichier manquant : {path_2ph}")

    data = np.load(path_2ph, allow_pickle=True).item()

    fluo         = np.asarray(data["fluo"],         dtype=float)[:, :-BUFFER_FRAMES]
    t_fluo_plane = np.asarray(data["t_fluo_plane"], dtype=float)[:, :-BUFFER_FRAMES]
    trial_matrix = data["trial_matrix"].copy()

    return {
        "group":        group,
        "mouse":        mouse,
        "spot":         spot,
        "day":          day,
        "fluo":         fluo,         # (n_neurons, n_frames)
        "t_fluo_plane": t_fluo_plane, # (5, n_frames)
        "trial_matrix": trial_matrix, # DataFrame (n_trials, 28)
    }


def extract_trials(session: dict,
                   pre_frames: int = 5,
                   post_frames: int = 10) -> tuple:
    """
    Découpe la fluo en fenêtres d'essais autour du son.

    Fenêtre : [idx_tone - pre_frames : idx_tone + post_frames]
    = 1s avant + 2s après le son à 5 Hz = 15 frames

    Exclut : early_press + essais hors limites

    Retourne :
        X : (n_trials, n_neurons, n_frames)  float64
        y : (n_trials,)                       int   1=hit, 0=miss
        trial_ids : (n_trials,)               int
    """
    fluo = session["fluo"]
    tm   = session["trial_matrix"]
    n_frames = fluo.shape[1]

    # Garder seulement hit et miss
    valid = tm[tm["name"].isin(["hit", "miss"])].copy()

    X, y, trial_ids = [], [], []

    for _, row in valid.iterrows():
        idx = int(row["idx_tone_0"])
        start = idx - pre_frames
        end   = idx + post_frames

        # Filtrer hors limites
        if start < 0 or end > n_frames:
            continue

        X.append(fluo[:, start:end])              # (n_neurons, n_frames)
        y.append(1 if row["name"] == "hit" else 0)
        trial_ids.append(int(row["trialID"]))

    if len(X) == 0:
        raise ValueError(f"Aucun essai valide pour "
                         f"{session['group']}/{session['mouse']}/"
                         f"{session['spot']}/D{session['day']}")

    return np.array(X), np.array(y), np.array(trial_ids)


def load_all_sessions(raw_dir: Path,
                      config: list = SESSIONS_CONFIG) -> list:
    """
    Charge toutes les sessions définies dans config.
    Retourne une liste de dicts avec fluo, trial_matrix, etc.
    """
    sessions = []
    for group, mouse, spot, days in config:
        for day in days:
            try:
                s = load_session(raw_dir, group, mouse, spot, day)
                sessions.append(s)
                print(f"✓ {group}/{mouse}/{spot}/D{day} — "
                      f"{s['fluo'].shape[0]} neurones, "
                      f"{s['trial_matrix']['name'].value_counts().to_dict()}")
            except FileNotFoundError as e:
                print(f"✗ {e}")
    return sessions


def minimum_neurons(sessions: list) -> int:
    """Retourne le nombre minimum de neurones parmi toutes les sessions."""
    return min(s["fluo"].shape[0] for s in sessions)