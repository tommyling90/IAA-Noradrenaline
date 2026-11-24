# Décoder l’apprentissage moteur dans le cortex, avec et sans noradrénaline 

Machine-learning project using experimental data related to noradrenaline.

Python 3.11 required.

## Project structure

```text
IAA-noradrenaline/
├── data/
│   ├── raw/                 # Original experimental data (not tracked by Git)
│   │   ├── ctrl/
│   │   │   └── 1387/
│   │   │       └── Spot 1/
│   │   │           ├── D1_lever.npy
│   │   │           └── D1_2PHData.npy
│   │   └── 1a_ko/
│   │       └── 1382/
│   │           └── Spot 2/
│   │               ├── D2_lever.npy
│   │               └── D2_2PHData.npy
│   └── processed/           # Generated/processed data (not tracked by Git)
├── notebooks/               # Data exploration and experiments
├── src/                     # Preprocessing and modeling code
├── results/                 # Evaluation results and figures
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create and activate a venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```
For Windows:
```bash
.venv\Scripts\activate
```

Install the required dependencies
```bash
python -m pip install -r requirements.txt
```

## Data setup

1. Download the experimental data folders shared by the team (e.g., `ctrl/` and `1a_ko/`) from OneDrive.
2. Place these folders inside **`data/raw/`**.
3. **Preserve the original subfolder structure and `.npy` filenames** (subject ID, spot, day, etc.).

Example path: `data/raw/ctrl/1387/Spot 1/D1_lever.npy`.

Raw experimental data should stay on your computer/shared drive and **must not be committed to GitHub**.

Run the `exploration.ipynb` code to make sure the data can be read.

## Workflow for Objective 1:
```bash
Notebook: extract_data
    ↓
Save raw fluo + lever_2p to .npz #these 2 steps are already executed by notebook and data saved into data/preprocessed/objectif1
    ↓
data_loader.py #only functions here
    ├── Load sessions
    └── Find minimum number of neurons
    ↓
preprocessing.py #only functions here
    ├── Average 5 lever positions (nanmean)
    ├── Sample 80% × min_neurons
    └── Construct temporal features (t ± 3)
    ↓
evaluation.py
    ├── 5-fold outer CV
    ├── 3-fold inner CV
    └── Train / predict / evaluate #all steps until here should be shared by all ML methods
           ↓
    normalization/methods.py # normalize AFTER the fold! To prevent training data leakage
    ├── standard
    └── none
           ↓
    models/
    ├── svr.py
    └── ridge.py
```
### Pour tester sur Objectif 1, exécuter

`python -m src.objectif1.run_experiments --models your_model_name --normalisation your_norm_method`.
Noter que l'argument `--models` permet une liste de modèles (e.g. `ridge svr`).

## Workflow for Objective 2:
```bash
Notebook: exploration_obj2
    ↓
data_loader.py #only functions here
    ├── Load sessions from data/raw/ (ctrl/ and 1a ko/)
    ├── Extract trials from trial_matrix (hit/miss only, exclude early_press)
    └── Find minimum number of neurons across all sessions
    ↓
preprocessing.py #only functions here
    ├── Sample 80% × min_neurons (20 random draws)
    ├── Flatten trials (n_trials, n_neurons × n_frames)
    └── Compute class weights (imbalanced hit/miss ratio)
    ↓
evaluation.py
    ├── 5-fold consecutive CV (no random split — adjacent trials are similar)
    ├── Class weighting at training time
    ├── Train / predict / evaluate
    └── 100 permutations for chance level
           ↓
    normalization/
    └── standardize AFTER the fold! To prevent data leakage
           ↓
    models/
    ├── svm_linear.py      # LinearSVC + CalibratedClassifierCV (Levy et al. 2020)
    ├── logistic_l2.py     # Logistic Regression L2 (probabilistic linear baseline)
    ├── logistic_lowrank.py # Low-rank logistic (TruncatedSVD + LR, reduces overfitting)
    └── svm_rbf_pca.py     # SVM RBF after PCA (non-linear boundary in reduced space)
```

### Pour tester sur Objectif 2, exécuter

`python -m src.objectif2.run_experiments --models svm_linear logistic_l2 logistic_lowrank svm_rbf_pca --n_draws 20 --n_perm 100 --normalisation standard`

Noter que l'argument `--models` permet une liste de modèles (e.g. `svm_linear logistic_l2`).

Résultats sauvegardés dans `results/objectif2/` sous format JSON.