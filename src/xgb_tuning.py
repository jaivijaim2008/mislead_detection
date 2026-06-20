"""
xgb_tuning.py — Missed-Lead Detector
Optuna hyperparameter tuning for XGBoost on the merged Kaggle dataset.
Optimizes for AUC on a stratified validation split.

"""

import os, sys, json, warnings
import pandas as pd
import numpy as np
import pickle

import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "..", "data", "leads_merged.csv")
MODELS = os.path.join(BASE, "..", "models")
OUT    = os.path.join(BASE, "..", "outputs")

# Import shared feature engineering from train_model
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from train_model import load_and_engineer


# ─────────────────────────────────────────────────────────
# 2. OPTUNA OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────
def create_objective(X_train, y_train):
    """Create an Optuna objective that closes over the training data."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
            "eval_metric": "logloss",
            "random_state": 42,
        }

        model = xgb.XGBClassifier(**params)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        return scores.mean()

    return objective


# ─────────────────────────────────────────────────────────
# 3. RUN TUNING
# ─────────────────────────────────────────────────────────
def run_tuning(n_trials=100):
    """Run Optuna hyperparameter search for XGBoost."""
    print("=" * 60)
    print("  XGBOOST HYPERPARAMETER TUNING (Optuna)")
    print("  Dataset: leads_merged.csv (13,740 rows)")
    print(f"  Trials: {n_trials}")
    print("=" * 60)

    # Load data (load_and_engineer returns X, y, df)
    X, y, _df = load_and_engineer(DATA)
    print(f"\n[data] Loaded {len(X)} samples, {X.shape[1]} features")
    print(f"[data] Class distribution: missed={y.sum()} ({y.mean():.1%}), "
          f"replied={len(y)-y.sum()} ({1-y.mean():.1%})")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="xgb_lead_scoring"
    )

    objective = create_objective(X_scaled, y)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, val in study.best_params.items():
        print(f"    {key:25s}: {val}")

    # Train best model on full data and evaluate
    best_params = study.best_params.copy()
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model = xgb.XGBClassifier(**best_params, n_jobs=-1)
    best_model.fit(X_tr, y_tr)

    y_pred = best_model.predict(X_te)
    y_probs = best_model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_probs)
    report = classification_report(y_te, y_pred, target_names=["Replied", "Missed"])

    print(f"\n  Final Test AUC (best params): {auc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    # Save best model and params
    with open(os.path.join(MODELS, "xgb_tuned.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    # Save tuning results
    results = {
        "best_auc": round(study.best_value, 4),
        "test_auc": round(auc, 4),
        "best_params": {k: v if not isinstance(v, (np.floating, np.integer))
                        else float(v) for k, v in study.best_params.items()},
        "n_trials": n_trials,
        "dataset": "leads_merged.csv",
        "n_samples": len(X),
    }
    with open(os.path.join(OUT, "xgb_tuning_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved tuned model -> {os.path.join(MODELS, 'xgb_tuned.pkl')}")
    print(f"  Saved results -> {os.path.join(OUT, 'xgb_tuning_results.json')}")

    return best_model, study


if __name__ == "__main__":
    run_tuning(n_trials=100)
