"""
Objectif 2 — run_experiments.py
Script principal pour lancer tous les modèles sur toutes les sessions.
"""
import argparse
import json
import numpy as np
from pathlib import Path

from src.objectif2.data_loader import load_all_sessions, extract_trials, minimum_neurons
from src.objectif2.preprocessing import (
    sample_neurons, flatten_trials, standardize, compute_class_weights
)
from src.objectif2.evaluation import consecutive_folds
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Sessions pairées : (early, late) = indices dans load_all_sessions
SESSION_PAIRS = [
    (0, 1),   # ctrl/1384/Spot1  D1 vs D13
    (2, 3),   # ko/1382/Spot1    D1 vs D13
    (4, 5),   # ko/1382/Spot2    D2 vs D12
    (6, 7),   # ko/1386/Spot1    D1 vs D13
    (8, 9),   # ko/1386/Spot2    D2 vs D14
    (10, 11), # ko/1385/Spot1    D1 vs D13
    (12, 13), # ko/1385/Spot2    D2 vs D14
]

MODELS = {
    "svm_linear": {
        "import": "src.objectif2.models.svm_linear",
        "fn":     "get_model",
        "params": {"C": 1.0},
    },
    "logistic_l2": {
        "import": "src.objectif2.models.logistic_l2",
        "fn":     "get_model",
        "params": {"C": 1.0},
    },
    "logistic_lowrank": {
        "import": "src.objectif2.models.logistic_lowrank",
        "fn":     "LowRankLogistic",
        "params": {"n_components": 10, "C": 1.0},
        "is_class": True,
    },
    "svm_rbf_pca": {
        "import": "src.objectif2.models.svm_rbf_pca",
        "fn":     "get_model",
        "params": {"n_components": 20, "C": 1.0},
    },
}


def get_model_instance(model_cfg, class_weight):
    import importlib
    mod = importlib.import_module(model_cfg["import"])
    fn  = getattr(mod, model_cfg["fn"])
    if model_cfg.get("is_class"):
        return fn(**model_cfg["params"], class_weight=class_weight)
    return fn(**model_cfg["params"], class_weight=class_weight)


def run_session(session, model_name, model_cfg,
                n_subsamples, n_draws, n_folds, n_perm, normalize, rng_seed):
    X, y, _ = extract_trials(session)
    rng = np.random.default_rng(rng_seed)
    folds = consecutive_folds(len(y), n_folds)

    scores_ba, scores_auc = [], []
    for _ in range(n_draws):
        fold_ba, fold_auc = [], []
        for tr_idx, te_idx in folds:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if len(np.unique(y_te)) < 2:
                continue
            X_tr_s, idx = sample_neurons(X_tr, n_subsamples, rng)
            X_te_s = X_te[:, idx, :]
            X_tr_f = flatten_trials(X_tr_s)
            X_te_f = flatten_trials(X_te_s)
            if normalize:
                X_tr_f, X_te_f = standardize(X_tr_f, X_te_f)
            cw = compute_class_weights(y_tr)
            model = get_model_instance(model_cfg, cw)
            model.fit(X_tr_f, y_tr)
            y_pred = model.predict(X_te_f)
            fold_ba.append(balanced_accuracy_score(y_te, y_pred))
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te_f)[:, 1]
                fold_auc.append(roc_auc_score(y_te, y_prob))
        if fold_ba:
            scores_ba.append(np.mean(fold_ba))
        if fold_auc:
            scores_auc.append(np.mean(fold_auc))

    # Permutations
    perm_ba = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        fold_perm = []
        for tr_idx, te_idx in folds:
            if len(np.unique(y_perm[te_idx])) < 2:
                continue
            X_tr_s, idx = sample_neurons(X[tr_idx], n_subsamples, rng)
            X_te_s = X[te_idx][:, idx, :]
            X_tr_f, X_te_f = flatten_trials(X_tr_s), flatten_trials(X_te_s)
            if normalize:
                X_tr_f, X_te_f = standardize(X_tr_f, X_te_f)
            cw = compute_class_weights(y_perm[tr_idx])
            m = get_model_instance(model_cfg, cw)
            m.fit(X_tr_f, y_perm[tr_idx])
            fold_perm.append(balanced_accuracy_score(
                y_perm[te_idx], m.predict(X_te_f)))
        if fold_perm:
            perm_ba.append(np.mean(fold_perm))

    return {
        "session":  f"{session['group']}/{session['mouse']}/{session['spot']}/D{session['day']}",
        "group":    session["group"],
        "mouse":    session["mouse"],
        "spot":     session["spot"],
        "day":      session["day"],
        "mean_ba":  float(np.mean(scores_ba)),
        "std_ba":   float(np.std(scores_ba)),
        "mean_auc": float(np.mean(scores_auc)) if scores_auc else None,
        "chance":   float(np.mean(perm_ba)) if perm_ba else None,
        "all_ba":   [float(x) for x in scores_ba],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",        nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--n_draws",       type=int,  default=20)
    parser.add_argument("--n_folds",       type=int,  default=5)
    parser.add_argument("--n_perm",        type=int,  default=500)
    parser.add_argument("--normalisation", type=str,  default="standard")
    parser.add_argument("--output_dir",    type=str,  default="results/objectif2")
    args = parser.parse_args()

    ROOT    = Path(__file__).resolve().parents[2]
    raw_dir = ROOT / "data" / "raw"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Chargement des sessions...")
    sessions = load_all_sessions(raw_dir)
    min_n    = minimum_neurons(sessions)
    n_sub    = int(0.8 * min_n)
    normalize = args.normalisation == "standard"
    print(f"Min neurones={min_n}, n_subsamples={n_sub}, normalize={normalize}")

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Modèle inconnu : {model_name}")
            continue
        model_cfg = MODELS[model_name]
        print(f"\n=== {model_name} ===")
        all_results = []

        for pair_idx, (early_i, late_i) in enumerate(SESSION_PAIRS):
            s_early = sessions[early_i]
            s_late  = sessions[late_i]
            print(f"  Paire {pair_idx+1}: D{s_early['day']} vs D{s_late['day']} "
                  f"({s_early['group']}/{s_early['mouse']}/{s_early['spot']})")

            for phase, session in [("early", s_early), ("late", s_late)]:
                r = run_session(session, model_name, model_cfg,
                                n_sub, args.n_draws, args.n_folds,
                                args.n_perm, normalize, rng_seed=42)
                r["phase"] = phase
                r["pair"]  = pair_idx
                all_results.append(r)
                print(f"    {phase}: BA={r['mean_ba']:.3f} ± {r['std_ba']:.3f} "
                      f"(chance={r['chance']:.3f})")

        out_file = out_dir / f"{model_name}_results.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Sauvegardé : {out_file}")

    print("\nTerminé !")


if __name__ == "__main__":
    main()