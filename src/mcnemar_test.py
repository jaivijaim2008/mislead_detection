"""
mcnemar_test.py — Missed-Lead Detector
McNemar's Test for statistical significance between classifier pairs (Course Unit V).

McNemar's test evaluates whether two classifiers produce significantly different
results on the same dataset. It uses a 2x2 contingency table of disagreements:

                    Classifier B
                    Correct  | Wrong
  Classifier A  Correct |  a    |   b
               Wrong    |  c    |   d

  McNemar's statistic: χ² = (b - c)² / (b + c)
  With Yates correction: χ² = (|b - c| - 1)² / (b + c)

  H0: Both classifiers have the same error rate.
  If p < 0.05, we reject H0 → classifiers differ significantly.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray, correction: bool = True) -> dict:
    """
    Perform McNemar's test between two classifiers.

    Args:
        y_true: Ground truth labels
        y_pred_a: Predictions from classifier A
        y_pred_b: Predictions from classifier B
        correction: Whether to apply Yates' correction for continuity

    Returns:
        dict with chi2 statistic, p-value, interpretation, and contingency table
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    # Contingency table
    # b: A correct, B wrong
    # c: A wrong, B correct
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    a = int(np.sum(correct_a & correct_b))   # Both right
    d = int(np.sum(~correct_a & ~correct_b)) # Both wrong

    contingency = np.array([[a, b], [c, d]])

    # McNemar's test
    if correction:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    else:
        chi2 = (b - c) ** 2 / (b + c) if (b + c) > 0 else 0

    # p-value from chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(chi2, df=1) if chi2 > 0 else 1.0

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        if b > c:
            interpretation = f"Classifier A is significantly better (p={p_value:.4f})"
        elif c > b:
            interpretation = f"Classifier B is significantly better (p={p_value:.4f})"
        else:
            interpretation = f"Significant difference detected (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "contingency_table": {
            "both_correct": a,
            "A_correct_B_wrong": b,
            "A_wrong_B_correct": c,
            "both_wrong": d,
        },
        "interpretation": interpretation,
        "significant": p_value < alpha,
        "alpha": alpha,
    }


def run_mcnemar_analysis(models: dict, X_test: np.ndarray, y_test: np.ndarray,
                         model_names: list = None) -> pd.DataFrame:
    """
    Run McNemar's test for all pairs of models.

    Args:
        models: dict of {name: fitted_model} with predict() method
        X_test: Test features (scaled)
        y_test: Test labels
        model_names: Optional list of model names to include

    Returns:
        DataFrame with pairwise McNemar's test results
    """
    if model_names is None:
        model_names = list(models.keys())

    # Get predictions for each model
    predictions = {}
    for name in model_names:
        if name in models:
            predictions[name] = models[name].predict(X_test)

    # Run pairwise tests
    results = []
    for name_a, name_b in combinations(predictions.keys(), 2):
        result = mcnemar_test(y_test, predictions[name_a], predictions[name_b])
        result["classifier_a"] = name_a
        result["classifier_b"] = name_b
        results.append(result)

    df = pd.DataFrame(results)
    # Keep contingency_table so callers (e.g. print_mcnemar_results) can display
    # the raw disagreement counts (A_correct_B_wrong, A_wrong_B_correct).
    df = df[["classifier_a", "classifier_b", "chi2", "p_value",
             "significant", "interpretation", "contingency_table"]]

    return df


def print_mcnemar_results(results_df: pd.DataFrame):
    """Pretty-print McNemar's test results."""
    print(f"\n{'='*80}")
    print("  McNEMAR'S TEST — Pairwise Classifier Comparison")
    print("  (Course Unit V — Statistical Significance Testing)")
    print(f"{'='*80}")

    for _, row in results_df.iterrows():
        sig_marker = "***" if row["significant"] else "   "
        print(f"\n  {row['classifier_a']:25s} vs {row['classifier_b']:25s}")
        print(f"    chi2={row['chi2']:.4f}  |  p={row['p_value']:.6f}  |  "
              f"{'SIGNIFICANT' if row['significant'] else 'not significant'} {sig_marker}")
        print(f"    {row['interpretation']}")
        ct = row.get("contingency_table", {})
        if isinstance(ct, dict):
            print(f"    Contingency: A right/B wrong = {ct.get('A_correct_B_wrong', '?')}, "
                  f"A wrong/B right = {ct.get('A_wrong_B_correct', '?')}")


if __name__ == "__main__":
    import pickle
    from sklearn.preprocessing import StandardScaler
    from train_model import load_and_engineer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    import xgboost as xgb

    print("=" * 80)
    print("  MISSED-LEAD DETECTOR — McNEMAR'S TEST ANALYSIS")
    print("  (Course Unit V — Evaluation & Statistical Testing)")
    print("=" * 80)

    # Load data
    DATA = os.path.join(BASE, "..", "data", "leads_merged.csv")
    if not os.path.exists(DATA):
        DATA = os.path.join(BASE, "..", "data", "leads.csv")
    print(f"\n  Data: {DATA}")

    X, y, df = load_and_engineer(DATA)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Train models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42,
                                       n_jobs=-1),
    }

    print("\n  Training models...")
    for name, model in models.items():
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        acc = (y_pred == y_te).mean()
        print(f"    {name:25s}: Accuracy = {acc:.4f}")

    # Run McNemar's test
    print("\n  Running McNemar's test for all classifier pairs...")
    results = run_mcnemar_analysis(models, X_te_s, y_te)
    print_mcnemar_results(results)

    # Save results
    out_path = os.path.join(OUT, "mcnemar_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    # Summary
    sig_count = results["significant"].sum()
    total_pairs = len(results)
    print(f"\n  SUMMARY: {sig_count}/{total_pairs} pairs show statistically "
          f"significant differences (alpha=0.05)")
    print(f"{'='*80}")
