"""
Objectif 2 — evaluation.py
CV 5-fold consécutif + inner CV + 20 tirages de neurones.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from src.objectif2.preprocessing import sample_neurons, flatten_trials, standardize, compute_class_weights


def consecutive_folds(n_trials, n_folds=5):
    """
    Découpe les essais en n_folds blocs consécutifs.
    Retourne une liste de (train_idx, test_idx).
    """
    indices = np.arange(n_trials)
    fold_size = n_trials // n_folds
    folds = []
    for k in range(n_folds):
        start = k * fold_size
        end   = start + fold_size if k < n_folds - 1 else n_trials
        test_idx  = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_idx, test_idx))
    return folds


def evaluate_session_pair(session_early, session_late,
                           model_cls, model_params,
                           n_subsamples=133, n_draws=20,
                           n_folds=5, n_perm=500,
                           normalize=True, rng_seed=42):
    """
    Evalue un modèle sur une paire de sessions (début vs fin).
    Retourne un dict de scores pour early et late.
    """
    from src.objectif2.data_loader import extract_trials

    results = {}

    for label, session in [("early", session_early), ("late", session_late)]:
        X, y, _ = extract_trials(session)
        scores_ba, scores_auc = [], []

        rng = np.random.default_rng(rng_seed)
        folds = consecutive_folds(len(y), n_folds)

        for draw in range(n_draws):
            fold_ba, fold_auc = [], []

            for train_idx, test_idx in folds:
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # Sous-échantillonnage des neurones
                X_tr_s, idx = sample_neurons(X_tr, n_subsamples, rng)
                X_te_s = X_te[:, idx, :]

                # Aplatir
                X_tr_f = flatten_trials(X_tr_s)
                X_te_f = flatten_trials(X_te_s)

                # Normalisation APRES le fold
                if normalize:
                    X_tr_f, X_te_f = standardize(X_tr_f, X_te_f)

                # Poids de classe
                cw = compute_class_weights(y_tr)

                # Entraînement
                model = model_cls(class_weight=cw, **model_params)
                model.fit(X_tr_f, y_tr)

                # Prédiction
                y_pred = model.predict(X_te_f)
                y_prob = model.predict_proba(X_te_f)[:, 1] \
                    if hasattr(model, "predict_proba") else None

                fold_ba.append(balanced_accuracy_score(y_te, y_pred))
                if y_prob is not None and len(np.unique(y_te)) > 1:
                    fold_auc.append(roc_auc_score(y_te, y_prob))

            scores_ba.append(np.mean(fold_ba))
            if fold_auc:
                scores_auc.append(np.mean(fold_auc))

        # Niveau de hasard par permutation
        perm_ba = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            fold_perm = []
            for train_idx, test_idx in folds:
                X_tr_s, idx = sample_neurons(X[train_idx], n_subsamples, rng)
                X_te_s = X[test_idx][:, idx, :]
                X_tr_f = flatten_trials(X_tr_s)
                X_te_f = flatten_trials(X_te_s)
                if normalize:
                    X_tr_f, X_te_f = standardize(X_tr_f, X_te_f)
                cw = compute_class_weights(y_perm[train_idx])
                m = model_cls(class_weight=cw, **model_params)
                m.fit(X_tr_f, y_perm[train_idx])
                fold_perm.append(balanced_accuracy_score(
                    y_perm[test_idx], m.predict(X_te_f)))
            perm_ba.append(np.mean(fold_perm))

        results[label] = {
            "balanced_accuracy": np.array(scores_ba),
            "auc_roc":           np.array(scores_auc),
            "chance_ba":         np.array(perm_ba),
            "mean_ba":           float(np.mean(scores_ba)),
            "mean_auc":          float(np.mean(scores_auc)) if scores_auc else None,
            "chance_mean":       float(np.mean(perm_ba)),
            "session":           f"{session['group']}/{session['mouse']}/{session['spot']}/D{session['day']}",
        }

    return results