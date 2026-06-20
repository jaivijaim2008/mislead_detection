"""
train_model.py — Missed-Lead Detector
Feature engineering + multi-model training + evaluation report.
Covers course Units I-III and V.
"""

import os, sys, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is on sys.path so sibling modules resolve correctly
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sklearn.model_selection      import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing        import StandardScaler, LabelEncoder
from sklearn.linear_model         import LogisticRegression
from sklearn.naive_bayes          import GaussianNB
from sklearn.tree                 import DecisionTreeClassifier
from sklearn.ensemble             import RandomForestClassifier, VotingClassifier
from sklearn.metrics              import (classification_report, confusion_matrix,
                                           roc_auc_score, ConfusionMatrixDisplay)
from sklearn.cluster              import KMeans
import xgboost as xgb
import pickle
from deep_learning import (LeadClassifier, train_deep_model,
                           save_dl_model, plot_training_history,
                           plot_dl_confusion_matrix, plot_dl_roc_curve,
                           DEVICE)

warnings.filterwarnings("ignore")

BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "..", "data")
MODELS    = os.path.join(BASE, "..", "models")
OUT       = os.path.join(BASE, "..", "outputs")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUT,    exist_ok=True)

# Prefer merged dataset (synthetic + Kaggle) when available
MERGED    = os.path.join(DATA_DIR, "leads_merged.csv")
SYNTHETIC = os.path.join(DATA_DIR, "leads.csv")
DATA      = MERGED if os.path.exists(MERGED) else SYNTHETIC

# Load tuned XGBoost params if available
XGB_TUNED_PARAMS_PATH = os.path.join(OUT, "xgb_tuning_results.json")
XGB_TUNED_PARAMS = None
if os.path.exists(XGB_TUNED_PARAMS_PATH):
    with open(XGB_TUNED_PARAMS_PATH) as _f:
        _tuning = json.load(_f)
        XGB_TUNED_PARAMS = _tuning.get("best_params")
        print(f"[train] Loaded tuned XGBoost params (AUC: {_tuning.get('best_auc', '?')})")

# ─────────────────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING  (Unit I + II)
# ─────────────────────────────────────────────────────────
def load_and_engineer(path: str):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df["channel_enc"] = le.fit_transform(df["channel"])
    intent_words = ["price", "buy", "interested", "demo", "quote", "available"]
    df["intent_score"] = df["message_text"].apply(
        lambda t: sum(w in str(t).lower() for w in intent_words)
    )
    df["is_business_hours"] = df["message_hour"].between(9, 18).astype(int)
    df["gap_bucket"] = pd.cut(
        df["response_gap_hrs"], bins=[0, 6, 12, 24, 9999], labels=[0, 1, 2, 3]
    ).astype(int)
    features = [
        "channel_enc", "message_length", "high_intent_flag",
        "prev_contacts", "response_gap_hrs", "intent_score",
        "is_business_hours", "gap_bucket", "message_hour"
    ]
    X = df[features]
    y = df["replied"].map({1: 0, 0: 1})   # 1 = missed (positive class)
    return X, y, df

# ─────────────────────────────────────────────────────────
# 2. UNSUPERVISED CLUSTERING  (Unit III — K-Means)
# ─────────────────────────────────────────────────────────
def cluster_leads(df: pd.DataFrame) -> pd.DataFrame:
    feats  = df[["response_gap_hrs", "intent_score", "prev_contacts"]].copy()
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(feats_scaled)
    cluster_means = df.groupby("cluster")["response_gap_hrs"].mean().sort_values(ascending=False)
    remap = {cluster_means.index[0]: "High-Intent-Missed",
             cluster_means.index[1]: "Low-Intent",
             cluster_means.index[2]: "Already-Converted"}
    df["segment"] = df["cluster"].map(remap)
    print("\n[cluster_leads] Customer segment distribution:")
    print(df["segment"].value_counts().to_string())
    return df

# ─────────────────────────────────────────────────────────
# 3. TRAIN MODELS  (Unit II — Supervised Learning)
# ─────────────────────────────────────────────────────────
def train(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Use tuned XGBoost params if available from Optuna
    xgb_params = {"eval_metric": "logloss", "random_state": 42, "n_jobs": -1}
    if XGB_TUNED_PARAMS:
        # Only merge hyperparams, don't overwrite fixed keys
        fixed = set(xgb_params.keys()) | {"use_label_encoder"}
        xgb_params.update({k: v for k, v in XGB_TUNED_PARAMS.items() if k not in fixed})
        print(f"[train] Using Optuna-tuned XGBoost params")

    models = {
        "LogisticRegression" : LogisticRegression(max_iter=1000, random_state=42),
        "NaiveBayes"         : GaussianNB(),
        "DecisionTree"       : DecisionTreeClassifier(max_depth=6, random_state=42),
        "RandomForest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost"            : xgb.XGBClassifier(**xgb_params),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        cv_scores = cross_val_score(m, X_tr_s, y_tr, cv=cv, scoring="f1")
        y_pred = m.predict(X_te_s)
        auc    = roc_auc_score(y_te, m.predict_proba(X_te_s)[:, 1])
        results[name] = {"model": m, "cv_f1_mean": cv_scores.mean(),
                         "cv_f1_std": cv_scores.std(), "test_auc": auc,
                         "y_pred": y_pred}
        print(f"  {name:22s} | CV-F1 {cv_scores.mean():.3f}+/-{cv_scores.std():.3f} | AUC {auc:.3f}")

    # Ensemble (Unit III) — RF + XGB + LR
    ensemble = VotingClassifier(
        estimators=[("rf", models["RandomForest"]),
                    ("xgb", models["XGBoost"]),
                    ("lr", models["LogisticRegression"])],
        voting="soft"
    )
    ensemble.fit(X_tr_s, y_tr)
    y_ens    = ensemble.predict(X_te_s)
    auc_ens  = roc_auc_score(y_te, ensemble.predict_proba(X_te_s)[:, 1])
    print(f"  {'Ensemble(RF+XGB+LR)':22s} | AUC {auc_ens:.3f}")

    # ── Deep Learning Model (PyTorch GPU) ──
    print(f"\n[train] Training Deep Learning model on {DEVICE}...")
    X_tr_dl, X_val_dl, y_tr_dl, y_val_dl = train_test_split(
        X_tr_s, y_tr.values, test_size=0.15, random_state=42, stratify=y_tr.values
    )
    # Scale epochs/batch to dataset size for efficiency
    n_samples = len(X_tr_dl)
    dl_epochs = min(200, max(50, n_samples // 20))
    dl_batch  = min(256, max(32, n_samples // 10))
    print(f"[train] DL config: {dl_epochs} epochs, batch_size={dl_batch} (n={n_samples})")
    dl_model, dl_history = train_deep_model(
        X_tr_dl, y_tr_dl, X_val_dl, y_val_dl,
        epochs=dl_epochs, lr=1e-3, batch_size=dl_batch,
        dropout=0.3, patience=20, verbose=True
    )
    y_dl_pred  = dl_model.predict_np(X_te_s)
    y_dl_probs = dl_model.predict_proba_np(X_te_s)[:, 1]
    auc_dl     = roc_auc_score(y_te, y_dl_probs)
    f1_dl      = classification_report(y_te, y_dl_pred, output_dict=True)["1"]["f1-score"]
    print(f"  {'DeepLearning(PYTorch)':22s} | AUC {auc_dl:.3f} | F1 {f1_dl:.3f}")

    # Save DL model artifacts
    save_dl_model(dl_model, scaler, dl_history, MODELS)
    plot_training_history(dl_history, OUT)
    plot_dl_confusion_matrix(y_te.values, y_dl_pred, OUT)
    plot_dl_roc_curve(y_te.values, y_dl_probs, OUT)

    # ── Grand Ensemble (ML + DL) — soft-voting with DL probabilities ──
    ml_ens_probs = ensemble.predict_proba(X_te_s)[:, 1]
    grand_ens_probs = 0.5 * ml_ens_probs + 0.5 * y_dl_probs
    grand_ens_pred  = (grand_ens_probs >= 0.5).astype(int)
    auc_grand      = roc_auc_score(y_te, grand_ens_probs)
    report_grand   = classification_report(y_te, grand_ens_pred,
                                            target_names=["Replied", "Missed"])
    print(f"\n  {'GrandEnsemble(ML+DL)':22s} | AUC {auc_grand:.3f}")
    print(f"\n[train] Grand Ensemble Classification Report:\n", report_grand)

    # Pick best model by AUC
    all_scores = {**{k: v["test_auc"] for k, v in results.items()},
                  "Ensemble(RF+XGB+LR)": auc_ens,
                  "DeepLearning(PyTorch)": auc_dl,
                  "GrandEnsemble(ML+DL)": auc_grand}
    print("\n[train] === MODEL COMPARISON ===")
    for name, auc_val in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:22s} | AUC {auc_val:.4f}")
    best_name = max(all_scores, key=all_scores.get)
    print(f"\n  >> Best model: {best_name} (AUC {all_scores[best_name]:.4f})")

    # Save the best ML model (not DL, which is saved separately)
    ml_only_names = [k for k in results]
    best_ml_name = max(ml_only_names, key=lambda k: results[k]["test_auc"])
    best_m = results[best_ml_name]["model"]
    print(f"[train] Best ML model: {best_ml_name} (AUC {results[best_ml_name]['test_auc']:.4f})")
    with open(os.path.join(MODELS, "best_model.pkl"), "wb") as f: pickle.dump(best_m, f)
    with open(os.path.join(MODELS, "scaler.pkl"),     "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(MODELS, "ensemble.pkl"),   "wb") as f: pickle.dump(ensemble, f)

    # Save grand ensemble scores + all comparison for dashboard
    comparison = {"models": {}, "best": best_name}
    for name, auc_val in all_scores.items():
        comparison["models"][name] = {"auc": round(auc_val, 4)}
    with open(os.path.join(OUT, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # Confusion matrix for grand ensemble
    cm = confusion_matrix(y_te, grand_ens_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Replied", "Missed"]).plot(ax=ax)
    ax.set_title("Grand Ensemble (ML+DL) - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Save ensemble classification report
    with open(os.path.join(OUT, "classification_report.txt"), "w") as f:
        f.write(report_grand)

    return best_m, scaler, ensemble, dl_model, X_te_s, y_te

# ─────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE  (Unit II — Interpretability)
# ─────────────────────────────────────────────────────────
def plot_feature_importance(model, X, fallback_model=None):
    plot_model = model
    title_suffix = "Best Model"
    if not hasattr(model, "feature_importances_"):
        if fallback_model and hasattr(fallback_model, "feature_importances_"):
            plot_model = fallback_model
            title_suffix = "Random Forest"
        else:
            print("[train] Skipping feature importance — model has no feature_importances_.")
            return
    fi = pd.Series(plot_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    fi.plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title(f"Feature Importances ({title_suffix})")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"[train] Feature importance plot saved (using {title_suffix}).")

if __name__ == "__main__":
    print("=" * 60)
    print("  MISSED-LEAD DETECTOR - MODEL TRAINING PIPELINE")
    print("  (Traditional ML + Deep Learning)")
    print("=" * 60)
    print(f"  Data source: {DATA}")
    X, y, df = load_and_engineer(DATA)
    df        = cluster_leads(df)
    df.to_csv(os.path.join(OUT, "leads_segmented.csv"), index=False)
    print("\n[train] Model comparison (5-fold stratified CV + DL):")
    best_m, scaler, ensemble, dl_model, X_te_s, y_te = train(X, y)
    # Use RF from the ensemble as fallback for feature importance
    rf_model = ensemble.named_estimators_.get("rf", None)
    plot_feature_importance(best_m, X, fallback_model=rf_model)
    print("\n[train] All artefacts saved to /models and /outputs")
