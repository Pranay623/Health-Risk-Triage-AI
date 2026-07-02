"""
train.py
=========
Training loop for the Phase 1 multi-model comparison.

Runs all 6 model families through:
  1. Stratified 5-fold cross-validation with RandomizedSearchCV
     for hyperparameter tuning (fit on train set only)
  2. Final refit on the full training set with best hyperparameters
  3. Saves each fitted model to models/ as a .joblib file
  4. Saves a CV results summary to reports/results_summary.md

TRAINING STRATEGY
------------------
  Cross-Validation:
    StratifiedKFold with k=5, stratified by UrgencyLevel.
    Stratification ensures each fold has representative proportions
    of all 4 urgency classes, including rare High/Critical rows.

  Hyperparameter Search:
    RandomizedSearchCV with n_iter=20 per model.
    Scoring metric: 'f1_macro' — macro-averaged F1 across all 4 classes.
    Why f1_macro and not accuracy or AUROC for CV selection?
      - Accuracy is dominated by the 87% Low class — a model that
        always predicts Low gets 87% accuracy and is clinically useless.
      - f1_macro weights all 4 classes equally regardless of frequency,
        so it penalises models that ignore High/Critical.
      - AUROC is computed in evaluate.py after final model selection,
        not used for hyperparameter search (avoids the AUROC illusion
        flagged in our literature review: JMIR 2021;23(2):e25187).

  Class Imbalance:
    sklearn models: class_weight='balanced' (set in models.py)
    XGBoost: sample_weight array passed to fit(), computed from
             preprocessing.py class weights mapped to each training row.

  Final Refit:
    After CV selects best hyperparameters, the model is refit on the
    FULL training set (not just 4/5 of it) before test evaluation.
    This is standard practice and gives the final model maximum data.

TRIPOD COMPLIANCE
------------------
  Item 10b : Random seed fixed at 42 throughout (matches preprocessing)
  Item 15  : Best hyperparameters logged and saved per model
  Item 16  : Test set evaluation kept separate (done in evaluate.py,
             NOT here — train.py never touches the test set)

OUTPUTS
--------
  models/<model_name>_best.joblib   — fitted best model per family
  reports/cv_results.json           — CV scores and best params per model
  Console: live progress per model  — estimated time, CV score

Usage
-----
    python src/train.py

    Or import:
    from src.train import run_training_pipeline
    results = run_training_pipeline(train_path, models_dir, reports_dir)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.models import get_all_models, get_all_param_grids, RANDOM_STATE
from src.preprocessing import TARGET

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ======================================================================
# CONSTANTS
# ======================================================================

CV_FOLDS    = 5          # StratifiedKFold folds
N_ITER      = 20         # RandomizedSearchCV iterations per model
CV_SCORING  = "f1_macro" # Primary CV selection metric (see docstring)
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")

FEATURE_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "ShockIndex", "PulsePressure", "NEWS2Score",
    "Age", "AgeGroup", "Gender", "ICULOS",
]

LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


# ======================================================================
# SAMPLE WEIGHT HELPER (for XGBoost)
# ======================================================================

def _make_sample_weights(y: pd.Series, class_weights: dict[int, float]) -> np.ndarray:
    """
    Convert per-class weights dict into a per-sample weight array.
    Used for XGBoost which accepts sample_weight in .fit() rather
    than a class_weight constructor parameter.

    Parameters
    ----------
    y             : Series of integer class labels (0-3)
    class_weights : dict {class_int: float} from preprocessing.py

    Returns
    -------
    np.ndarray of shape (n_samples,) with each sample's weight
    """
    return np.array([class_weights[label] for label in y])


# ======================================================================
# SINGLE MODEL TRAINING
# ======================================================================

def train_single_model(
    name: str,
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict[int, float],
) -> tuple[object, dict]:
    """
    Run RandomizedSearchCV for one model, then refit best on full train.

    Parameters
    ----------
    name          : model name string (e.g. 'xgboost')
    model         : unfitted sklearn-compatible estimator
    param_grid    : hyperparameter search space dict
    X_train       : training features DataFrame
    y_train       : training labels Series (UrgencyLevel 0-3)
    class_weights : precomputed class weight dict from preprocessing.py

    Returns
    -------
    best_model : fitted estimator with best hyperparameters
    result     : dict with CV scores and best params for logging
    """
    logger.info(f"  Training {name} ...")
    t_start = time.time()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # XGBoost needs sample_weight via fit_params, not class_weight param
    fit_params = {}
    if name == "xgboost":
        sample_weights = _make_sample_weights(y_train, class_weights)
        fit_params["sample_weight"] = sample_weights

    # Skip RandomizedSearchCV if grid is empty (use model defaults)
    if not param_grid:
        logger.info(f"  No param grid for {name} — fitting with defaults.")
        if fit_params:
            model.fit(X_train, y_train, **fit_params)
        else:
            model.fit(X_train, y_train)
        best_model  = model
        best_params = {}
        cv_mean     = None
        cv_std      = None
    else:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=N_ITER,
            scoring=CV_SCORING,
            cv=cv,
            refit=True,           # refit best params on full train set
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
            error_score="raise",
        )

        if fit_params:
            # For XGBoost: pass sample_weight through fit_params
            # RandomizedSearchCV propagates fit_params to each fold's fit()
            search.fit(X_train, y_train, **fit_params)
        else:
            search.fit(X_train, y_train)

        best_model  = search.best_estimator_
        best_params = search.best_params_
        cv_mean     = search.cv_results_["mean_test_score"][search.best_index_]
        cv_std      = search.cv_results_["std_test_score"][search.best_index_]

    elapsed = time.time() - t_start

    result = {
        "model"       : name,
        "best_params" : best_params,
        "cv_f1_macro_mean": round(cv_mean, 4) if cv_mean is not None else None,
        "cv_f1_macro_std" : round(cv_std,  4) if cv_std  is not None else None,
        "train_time_sec"  : round(elapsed, 1),
    }

    logger.info(
        f"  {name:<25} CV f1_macro: "
        f"{cv_mean:.4f} +/- {cv_std:.4f}  "
        f"[{elapsed:.1f}s]"
        if cv_mean is not None
        else f"  {name:<25} fitted (no CV search)  [{elapsed:.1f}s]"
    )

    return best_model, result


# ======================================================================
# FULL PIPELINE
# ======================================================================

def run_training_pipeline(
    train_path    : str | Path = "data/processed/phase1/phase1_train.parquet",
    models_dir    : str | Path = MODELS_DIR,
    reports_dir   : str | Path = REPORTS_DIR,
) -> dict[str, dict]:
    """
    Run the full training pipeline for all 6 model families.

    Parameters
    ----------
    train_path  : path to phase1_train.parquet
    models_dir  : directory to save fitted model .joblib files
    reports_dir : directory to save CV results JSON

    Returns
    -------
    cv_results : dict {model_name: result_dict} with CV scores,
                 best params, and training times for all models
    """
    train_path  = Path(train_path)
    models_dir  = Path(models_dir)
    reports_dir = Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
    logger.info(f"Loading training data from {train_path} ...")
    train_df = pd.read_parquet(train_path)

    missing_features = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing_features:
        raise ValueError(f"Training data missing features: {missing_features}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]

    logger.info(f"  X_train shape : {X_train.shape}")
    logger.info(f"  y_train dist  : {y_train.value_counts().sort_index().to_dict()}")

    # ------------------------------------------------------------------
    # Load class weights from preprocessor artifact
    # ------------------------------------------------------------------
    preprocessor_path = Path("data/processed/phase1/preprocessor.joblib")
    if preprocessor_path.exists():
        preprocessor  = joblib.load(preprocessor_path)
        # Recompute class weights from y_train (most reliable source)
    
    # Compute class weights directly from training labels
    from src.preprocessing import compute_class_weights
    class_weights = compute_class_weights(y_train)
    logger.info(f"  Class weights : {class_weights}")

    # ------------------------------------------------------------------
    # Get model registry
    # ------------------------------------------------------------------
    all_models      = get_all_models()
    all_param_grids = get_all_param_grids()

    # ------------------------------------------------------------------
    # Train all models
    # ------------------------------------------------------------------
    cv_results    = {}
    fitted_models = {}

    logger.info(f"\nTraining {len(all_models)} models with "
                f"{CV_FOLDS}-fold CV, {N_ITER} random search iterations each...\n")

    total_start = time.time()

    for name, model in all_models.items():
        param_grid = all_param_grids.get(name, {})

        try:
            best_model, result = train_single_model(
                name=name,
                model=model,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                class_weights=class_weights,
            )

            # Save fitted model artifact
            model_path = models_dir / f"{name}_best.joblib"
            joblib.dump(best_model, model_path)
            logger.info(f"  Saved -> {model_path}")

            result["model_path"] = str(model_path)
            cv_results[name]     = result
            fitted_models[name]  = best_model

        except Exception as e:
            logger.error(f"  [FAILED] {name}: {e}")
            cv_results[name] = {"model": name, "error": str(e)}

    total_elapsed = time.time() - total_start
    logger.info(f"\nAll models trained in {total_elapsed:.1f}s total.")

    # ------------------------------------------------------------------
    # Save CV results to JSON
    # ------------------------------------------------------------------
    cv_path = reports_dir / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.info(f"CV results saved -> {cv_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    _print_cv_summary(cv_results)

    return cv_results


# ======================================================================
# SUMMARY TABLE
# ======================================================================

def _print_cv_summary(cv_results: dict) -> None:
    """Print a formatted CV results comparison table."""
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON — Cross-Validation Results (f1_macro)")
    print("=" * 65)
    print(f"  {'Model':<25} {'CV f1_macro':>12}  {'+/-':>8}  {'Time':>8}")
    print("-" * 65)

    # Sort by CV score descending
    sorted_results = sorted(
        cv_results.items(),
        key=lambda x: x[1].get("cv_f1_macro_mean") or 0,
        reverse=True,
    )

    for name, result in sorted_results:
        if "error" in result:
            print(f"  {name:<25} {'ERROR':>12}  {'':>8}  {'':>8}")
            continue

        mean = result.get("cv_f1_macro_mean")
        std  = result.get("cv_f1_macro_std")
        t    = result.get("train_time_sec", 0)

        mean_str = f"{mean:.4f}" if mean is not None else "N/A"
        std_str  = f"{std:.4f}"  if std  is not None else "N/A"
        time_str = f"{t:.1f}s"

        print(f"  {name:<25} {mean_str:>12}  {std_str:>8}  {time_str:>8}")

    print("=" * 65)
    print("  Next step: run evaluate.py for test-set metrics + SHAP")
    print("=" * 65 + "\n")


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":
    run_training_pipeline()