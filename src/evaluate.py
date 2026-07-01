"""
evaluate.py
============
Test-set evaluation and explainability for the Phase 1 model comparison.

Loads each fitted model from models/ and evaluates it on the held-out
test set (phase1_test.parquet), producing:

  Per-model metrics:
    - Classification report (precision, recall, F1 per class)
    - Macro and weighted F1
    - Multi-class AUROC (one-vs-rest)
    - Confusion matrix

  Cross-model comparison:
    - Summary table: all models ranked by test F1-macro
    - Critical-class recall comparison (clinically most important metric)

  Explainability (SHAP) for the best model:
    - SHAP values computed for the test set
    - Mean absolute SHAP per feature (global importance)
    - Saved to reports/figures/ as a text-based ranking

  All results saved to:
    reports/test_results.json       — full metrics per model
    reports/results_summary.md      — human-readable final report
    reports/figures/                — confusion matrices + SHAP output

METRIC PHILOSOPHY
------------------
  Primary metric for ranking  : F1-macro (equal weight to all 4 classes)
  Most important clinical metric: Recall for class 3 (Critical)

  Why Critical recall matters most:
    A missed Critical case (FN: predicted Low/Medium when actually
    Critical) is the highest-consequence error in a triage system —
    a patient who needed immediate intervention didn't get flagged.
    This maps directly to the "sensitivity for rare critical events"
    concern flagged in our EWS literature review (JMIR 2021).

    Conversely, a false Critical alarm (FP: predicted Critical when
    actually Low) causes alert fatigue — also documented as a key
    failure mode. So we report BOTH recall and precision for Critical.

  AUROC:
    Reported for completeness and literature comparability, but NOT
    used as the primary ranking metric. Per our literature review,
    AUROC can be misleadingly high when events are rare (a model that
    scores Critical patients slightly higher than Low patients gets
    high AUROC even with poor precision). F1-macro is more honest here.

TRIPOD COMPLIANCE
------------------
  Item 10 : Test set evaluation on data never seen during training
  Item 15  : Full model specification saved in models/ artifacts
  Item 16  : Performance reported per class, not just overall accuracy

Usage
-----
    python -m src.evaluate

    Or import:
    from src.evaluate import run_evaluation_pipeline
    results = run_evaluation_pipeline()
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ======================================================================
# CONSTANTS
# ======================================================================

FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "ShockIndex", "PulsePressure", "NEWS2Score",
    "Age", "AgeGroup", "Gender", "ICULOS",
]

TARGET      = "UrgencyLevel"
LABEL_NAMES = ["Low", "Medium", "High", "Critical"]
LABEL_MAP   = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"


# ======================================================================
# LOAD TEST DATA AND MODELS
# ======================================================================

def load_test_data(test_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load test features and labels."""
    df = pd.read_parquet(test_path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Test data missing features: {missing}")
    X_test = df[FEATURE_COLS]
    y_test = df[TARGET]
    return X_test, y_test


def load_fitted_models(models_dir: Path) -> dict[str, object]:
    """Load all saved .joblib model files from models directory."""
    models = {}
    for path in sorted(models_dir.glob("*_best.joblib")):
        name = path.stem.replace("_best", "")
        models[name] = joblib.load(path)
        logger.info(f"  Loaded: {name} <- {path}")
    if not models:
        raise FileNotFoundError(
            f"No fitted models found in {models_dir}. Run train.py first."
        )
    return models


# ======================================================================
# METRICS COMPUTATION
# ======================================================================

def evaluate_model(
    name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute full evaluation metrics for one model on the test set.

    Returns
    -------
    dict with:
      f1_macro         : float
      f1_weighted      : float
      auroc_macro      : float  (one-vs-rest, macro averaged)
      per_class        : dict {class_name: {precision, recall, f1, support}}
      confusion_matrix : list of lists (raw counts)
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)   # shape (n, 4)

    # --- Core metrics ------------------------------------------------
    f1_macro    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # AUROC — one-vs-rest, macro averaged across 4 classes
    try:
        auroc = roc_auc_score(
            y_test, y_proba,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auroc = None   # can fail if a class has no positive examples in test

    # --- Per-class metrics -------------------------------------------
    report = classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls_name in LABEL_NAMES:
        per_class[cls_name] = {
            "precision": round(report[cls_name]["precision"], 4),
            "recall"   : round(report[cls_name]["recall"],    4),
            "f1"       : round(report[cls_name]["f1-score"],  4),
            "support"  : int(report[cls_name]["support"]),
        }

    # --- Confusion matrix -------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    result = {
        "model"           : name,
        "f1_macro"        : round(f1_macro,    4),
        "f1_weighted"     : round(f1_weighted, 4),
        "auroc_macro"     : round(auroc, 4) if auroc is not None else None,
        "per_class"       : per_class,
        "confusion_matrix": cm.tolist(),
    }

    return result


# ======================================================================
# CONFUSION MATRIX PRINTER
# ======================================================================

def print_confusion_matrix(name: str, cm: list[list[int]]) -> None:
    """Print a readable confusion matrix to console."""
    print(f"\n  Confusion Matrix — {name}")
    print(f"  {'':12}", end="")
    for label in LABEL_NAMES:
        print(f"  {label:>10}", end="")
    print(f"\n  {'-'*56}")
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:12}", end="")
        for j, val in enumerate(row):
            marker = " *" if (i == j) else "  "
            print(f"  {val:>8}{marker}", end="")
        print()


# ======================================================================
# SHAP EXPLAINABILITY
# ======================================================================

def compute_shap_importance(
    model,
    X_test: pd.DataFrame,
    model_name: str,
    figures_dir: Path,
) -> dict[str, float] | None:
    """
    Compute SHAP feature importance for tree-based and linear models.

    Uses TreeExplainer for RF/XGBoost/DT (fast, exact).
    Uses LinearExplainer for LogisticRegression (fast, exact).
    Uses KernelExplainer for MLP as fallback (slow — uses a subsample).

    Returns dict {feature_name: mean_abs_shap} or None if SHAP fails.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None

    logger.info(f"  Computing SHAP values for {model_name} ...")

    try:
        # Tree-based models — fast exact TreeExplainer
        if model_name in ("xgboost", "random_forest", "decision_tree"):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            # Handle 3D array (n_samples, n_features, n_classes) or list of arrays
            if isinstance(shap_values, list):
                abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
            elif len(np.shape(shap_values)) == 3:
                # Average across classes (axis 2) then samples (axis 0)
                abs_shap = np.abs(shap_values).mean(axis=2).mean(axis=0)
            else:
                abs_shap = np.abs(shap_values).mean(axis=0)

        # Linear model
        elif model_name == "logistic_regression":
            explainer   = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                abs_shap = np.mean(
                    [np.abs(sv) for sv in shap_values], axis=0
                ).mean(axis=0)
            else:
                abs_shap = np.abs(shap_values).mean(axis=0)

        # Neural / other — KernelExplainer on a small subsample
        else:
            background  = shap.sample(X_test, 50, random_state=42)
            explainer   = shap.KernelExplainer(model.predict_proba, background)
            sample      = X_test.iloc[:100]
            shap_values = explainer.shap_values(sample)
            if isinstance(shap_values, list):
                abs_shap = np.mean(
                    [np.abs(sv) for sv in shap_values], axis=0
                ).mean(axis=0)
            else:
                abs_shap = np.abs(shap_values).mean(axis=0)

        importance = dict(zip(FEATURE_COLS, abs_shap))
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        # Save SHAP ranking to text file
        shap_path = figures_dir / f"shap_{model_name}.txt"
        with open(shap_path, "w") as f:
            f.write(f"SHAP Feature Importance — {model_name}\n")
            f.write("=" * 45 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<20} {'Mean |SHAP|':>12}\n")
            f.write("-" * 45 + "\n")
            for rank, (feat, val) in enumerate(importance_sorted.items(), 1):
                bar = "#" * int(val / max(importance_sorted.values()) * 20)
                f.write(f"{rank:<6} {feat:<20} {val:>12.4f}  {bar}\n")

        logger.info(f"  SHAP saved -> {shap_path}")
        return importance_sorted

    except Exception as e:
        logger.warning(f"  SHAP failed for {model_name}: {e}")
        return None


# ======================================================================
# RESULTS SUMMARY REPORT
# ======================================================================

def save_results_summary(
    all_results: dict[str, dict],
    shap_results: dict[str, dict],
    reports_dir: Path,
) -> None:
    """Write a human-readable markdown results summary."""
    summary_path = reports_dir / "results_summary.md"

    # Sort by f1_macro descending
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("f1_macro", 0),
        reverse=True,
    )

    lines = [
        "# Health Risk Triage — Phase 1 Model Comparison Results\n",
        "## Overview\n",
        "Dataset: PhysioNet 2019 Sepsis Challenge (500-patient test slice)\n",
        "Task: 4-class urgency classification (Low / Medium / High / Critical)\n",
        "Label derivation: NEWS2-style vital scoring + SepsisLabel override\n",
        "Features: 14 (vitals + engineered: ShockIndex, PulsePressure, AgeGroup, NEWS2Score)\n",
        "Train/Test split: 80/20 stratified by UrgencyLevel\n",
        "Class imbalance: handled via class_weight='balanced'\n\n",
        "## Summary Table\n",
        "| Model | F1-Macro | F1-Weighted | AUROC | Critical Recall | Critical Precision |\n",
        "|-------|----------|-------------|-------|-----------------|--------------------|\n",
    ]

    for name, result in sorted_results:
        f1m  = result.get("f1_macro",    "N/A")
        f1w  = result.get("f1_weighted", "N/A")
        auc  = result.get("auroc_macro", "N/A")
        crit = result.get("per_class",   {}).get("Critical", {})
        cr   = crit.get("recall",    "N/A")
        cp   = crit.get("precision", "N/A")
        lines.append(
            f"| {name} | {f1m} | {f1w} | {auc} | {cr} | {cp} |\n"
        )

    lines.append("\n## Per-Model Details\n")
    for name, result in sorted_results:
        lines.append(f"### {name}\n")
        lines.append(f"- **F1-Macro**: {result.get('f1_macro')}\n")
        lines.append(f"- **AUROC**: {result.get('auroc_macro')}\n")
        lines.append("\n**Per-Class Metrics:**\n")
        lines.append("| Class | Precision | Recall | F1 | Support |\n")
        lines.append("|-------|-----------|--------|----|---------|\n")
        for cls, metrics in result.get("per_class", {}).items():
            lines.append(
                f"| {cls} | {metrics['precision']} | {metrics['recall']} "
                f"| {metrics['f1']} | {metrics['support']} |\n"
            )
        lines.append("\n")

    if shap_results:
        lines.append("## SHAP Feature Importance (Best Model)\n")
        best_model_name = sorted_results[0][0]
        shap = shap_results.get(best_model_name, {})
        if shap:
            lines.append(f"Model: **{best_model_name}**\n\n")
            lines.append("| Rank | Feature | Mean |SHAP| |\n")
            lines.append("|------|---------|--------|\n")
            for rank, (feat, val) in enumerate(shap.items(), 1):
                lines.append(f"| {rank} | {feat} | {val:.4f} |\n")

    lines.append("\n## Notes\n")
    lines.append("- SVM excluded: RBF kernel + Platt scaling ran >2.5h on 15k rows (O(n^2) complexity).\n")
    lines.append("- Row-level train/test split used (not patient-level). Patient-level split is a Phase 2 improvement.\n")
    lines.append("- Labels derived from NEWS2 thresholds + SepsisLabel override. See labeling.py for full documentation.\n")
    lines.append("- TRIPOD+AI reporting checklist applied throughout. See individual src/ files for Item references.\n")

    with open(summary_path, "w") as f:
        f.writelines(lines)

    logger.info(f"Results summary saved -> {summary_path}")


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_evaluation_pipeline(
    test_path  : str | Path = "data/processed/phase1/phase1_test.parquet",
    models_dir : str | Path = MODELS_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    run_shap_for: str = "xgboost",   # which model to run SHAP on
) -> dict[str, dict]:
    """
    Run full test-set evaluation for all fitted models.

    Parameters
    ----------
    test_path    : path to phase1_test.parquet
    models_dir   : directory containing *_best.joblib files
    reports_dir  : directory to save results
    run_shap_for : model name to run SHAP on (default: best model)
                   Set to 'all' to run SHAP for every model (slow for MLP)

    Returns
    -------
    all_results : dict {model_name: metrics_dict}
    """
    models_dir  = Path(models_dir)
    reports_dir = Path(reports_dir)
    figures_dir = Path(FIGURES_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load test data and fitted models
    # ------------------------------------------------------------------
    logger.info(f"Loading test data from {test_path} ...")
    X_test, y_test = load_test_data(test_path)
    logger.info(f"  Test set: {len(X_test):,} rows")
    logger.info(f"  Label distribution: {y_test.value_counts().sort_index().to_dict()}")

    logger.info(f"Loading fitted models from {models_dir} ...")
    fitted_models = load_fitted_models(models_dir)

    # ------------------------------------------------------------------
    # Evaluate all models
    # ------------------------------------------------------------------
    all_results = {}
    logger.info(f"\nEvaluating {len(fitted_models)} models on test set...\n")

    for name, model in fitted_models.items():
        logger.info(f"  Evaluating {name} ...")
        result = evaluate_model(name, model, X_test, y_test)
        all_results[name] = result
        print_confusion_matrix(name, result["confusion_matrix"])

    # ------------------------------------------------------------------
    # Print final comparison table
    # ------------------------------------------------------------------
    _print_test_summary(all_results)

    # ------------------------------------------------------------------
    # SHAP explainability
    # ------------------------------------------------------------------
    shap_results = {}
    shap_targets = (
        list(fitted_models.keys()) if run_shap_for == "all"
        else [run_shap_for] if run_shap_for in fitted_models
        else [sorted(all_results.items(),
                     key=lambda x: x[1].get("f1_macro", 0),
                     reverse=True)[0][0]]
    )

    for shap_model_name in shap_targets:
        shap_imp = compute_shap_importance(
            model=fitted_models[shap_model_name],
            X_test=X_test,
            model_name=shap_model_name,
            figures_dir=figures_dir,
        )
        if shap_imp:
            shap_results[shap_model_name] = shap_imp
            print(f"\n  SHAP Top Features — {shap_model_name}:")
            for rank, (feat, val) in enumerate(
                list(shap_imp.items())[:7], 1
            ):
                bar = "#" * int(val / list(shap_imp.values())[0] * 25)
                print(f"    {rank}. {feat:<20} {val:.4f}  {bar}")

    # ------------------------------------------------------------------
    # Save all outputs
    # ------------------------------------------------------------------
    results_path = reports_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nTest results saved -> {results_path}")

    save_results_summary(all_results, shap_results, reports_dir)

    return all_results


# ======================================================================
# SUMMARY TABLE PRINTER
# ======================================================================

def _print_test_summary(all_results: dict) -> None:
    """Print final ranked test-set comparison table."""
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("f1_macro", 0),
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("  FINAL TEST-SET RESULTS")
    print("=" * 80)
    print(f"  {'Model':<25} {'F1-Macro':>10} {'F1-Wt':>8} "
          f"{'AUROC':>8} {'Crit-R':>8} {'Crit-P':>8}")
    print("-" * 80)

    for name, result in sorted_results:
        f1m   = result.get("f1_macro",    0)
        f1w   = result.get("f1_weighted", 0)
        auc   = result.get("auroc_macro")
        crit  = result.get("per_class", {}).get("Critical", {})
        cr    = crit.get("recall",    0)
        cp    = crit.get("precision", 0)
        auc_s = f"{auc:.4f}" if auc else "  N/A"

        print(f"  {name:<25} {f1m:>10.4f} {f1w:>8.4f} "
              f"{auc_s:>8} {cr:>8.4f} {cp:>8.4f}")

    print("=" * 80)
    print("  Crit-R = Critical class Recall  (missed critical cases = dangerous)")
    print("  Crit-P = Critical class Precision (false critical alarms = alert fatigue)")
    print("=" * 80 + "\n")


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":
    run_evaluation_pipeline()
