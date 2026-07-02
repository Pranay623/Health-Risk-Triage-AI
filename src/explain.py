"""
explain.py
===========
Class-specific explainability for the Health Risk Triage project.

While evaluate.py provides global SHAP importance (averaged across
all classes), clinical users need to know exactly which features
drive a specific prediction. In triage, we care most about what
drives a patient into the 'Critical' category vs 'Low' category.

This script computes SHAP values using the best model (XGBoost)
and breaks down feature importance PER CLASS.

Outputs:
  reports/figures/shap_per_class.txt  — Text summary of top drivers per class
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ======================================================================
# CONSTANTS
# ======================================================================

FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "ShockIndex", "PulsePressure", "NEWS2Score",
    "Age", "AgeGroup", "Gender", "ICULOS",
]

TARGET = "UrgencyLevel"
LABEL_NAMES = ["Low", "Medium", "High", "Critical"]

MODEL_PATH = Path("models/xgboost_best.joblib")
TEST_PATH  = Path("data/processed/phase1/phase1_test.parquet")
OUT_FILE   = Path("reports/figures/shap_per_class.txt")


def run_explainability():
    if not MODEL_PATH.exists() or not TEST_PATH.exists():
        logger.error("Missing model or test data. Run train.py and evaluate.py first.")
        return

    logger.info(f"Loading best model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    logger.info(f"Loading test data from {TEST_PATH}")
    X_test = pd.read_parquet(TEST_PATH)[FEATURE_COLS]
    
    logger.info("Computing SHAP values (this may take a few seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP values for XGBoost 0.52.0 is a 3D array: (n_samples, n_features, n_classes)
    # Ensure it's the shape we expect
    if isinstance(shap_values, list):
        # Fallback if older SHAP: list of length n_classes, each (n_samples, n_features)
        shap_values_3d = np.stack(shap_values, axis=2)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_3d = shap_values
    else:
        logger.error(f"Unexpected SHAP values format: {type(shap_values)}")
        return
        
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUT_FILE, "w") as f:
        f.write("SHAP Feature Importance (Per Class)\n")
        f.write("=====================================\n\n")
        
        for i, class_name in enumerate(LABEL_NAMES):
            # Extract SHAP values for this specific class: (n_samples, n_features)
            class_shap = shap_values_3d[:, :, i]
            
            # Compute mean absolute SHAP for each feature for this class
            mean_abs_shap = np.abs(class_shap).mean(axis=0)
            
            # Sort features by importance
            importance = dict(zip(FEATURE_COLS, mean_abs_shap))
            importance_sorted = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            f.write(f"Class: {class_name}\n")
            f.write("-" * 45 + "\n")
            f.write(f"{'Rank':<6} {'Feature':<20} {'Mean |SHAP|':>12}\n")
            f.write("-" * 45 + "\n")
            
            for rank, (feat, val) in enumerate(importance_sorted.items(), 1):
                # Normalize bar length relative to the top feature for this class
                max_val = max(importance_sorted.values())
                bar = "#" * int((val / max_val) * 20) if max_val > 0 else ""
                f.write(f"{rank:<6} {feat:<20} {val:>12.4f}  {bar}\n")
            f.write("\n")
            
            # Also print to console
            print(f"\nTop drivers for {class_name}:")
            for rank, (feat, val) in enumerate(list(importance_sorted.items())[:5], 1):
                print(f"  {rank}. {feat:<15} ({val:.4f})")

    logger.info(f"Full per-class breakdown saved to {OUT_FILE}")

if __name__ == "__main__":
    run_explainability()
