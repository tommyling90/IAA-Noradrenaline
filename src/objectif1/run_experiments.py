import argparse
import csv
from pathlib import Path
import numpy as np
from .data_loader import list_all_paths, load_session, minimum_neurons
from .preprocessing import average_lever, sample_neurons, temporal_features
from .evaluation import evaluate_model
from .models import ridge, svr

SEED = 42

def run(args):



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", choices=["ridge", "svr"], default=["ridge", "svr"])
    p.add_argument("--normalization", choices=["standard", "none"], default="standard")
    run(p.parse_args())