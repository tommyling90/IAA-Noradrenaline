import argparse
import csv
from pathlib import Path
import numpy as np
from .data_loader import list_all_paths, load_session, minimum_neurons
from .preprocessing import average_lever, sample_neurons, temporal_features
from .evaluation import evaluate_model
from .models import ridge, svr

SEED = 42
FRACTION = 0.8
TIRAGE = 20

def run(args):
    root = Path(__file__).resolve().parents[2]
    paths = list_all_paths(root / "data" / "preprocessed" / "objectif1")
    n_min = minimum_neurons(paths)
    n_sample = int(np.floor(FRACTION * n_min))
    print(f"{len(paths)} sessions; min neurons={n_min}; sample per tirage={n_sample}")
    results = []

    # PAR SESSION
    for path in paths:
        fluo, lever_raw = load_session(path)
        y_all = average_lever(lever_raw)

        # PAR TIRAGE
        for tirage in range(TIRAGE):
            rng = np.random.default_rng(SEED + tirage)
            sampled, _ = sample_neurons(fluo, n_sample, rng)
            X, y, centers = temporal_features(sampled, y_all)

            # PAR MODÈLE
            for model_name in args.models:
                #TODO: add your own model name here
                estimator, grid = {"ridge": ridge.build, "svr": svr.build}[model_name]()
                result = evaluate_model(X, y, centers, estimator, grid, normalisation=args.normalisation)
                row = {"session": path.stem, "tirage": tirage + 1, # (tirage indexé sur 1)
                       "model": model_name, "n_neurons": n_sample,
                       "n_valid_frames": len(y), "r2": result["r2"],
                       "correlation": result["correlation"],
                       "best_params_per_test_block": repr(result["best_params_per_test_block"])}
                print(row["session"], row["tirage"], model_name, "R2", round(row["r2"], 4))
                results.append(row)

    output = root / "results"/ "objectif1" / "scores.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=["ridge", "svr"], default=["ridge", "svr"])
    p.add_argument("--normalisation", choices=["standard", "none"], default="standard")
    run(p.parse_args())