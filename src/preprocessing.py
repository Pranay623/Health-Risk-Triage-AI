"""
preprocessing.py
=================
Final data preparation stage for the Health Risk Triage project.

Takes the feature-engineered DataFrame from features.py and produces
model-ready train/test splits with:
  - Three-tier imputation for remaining missing vitals
  - StandardScaler for continuous features
  - Ordinal encoding confirmation for categorical features
  - Stratified train/test split preserving UrgencyLevel class ratios
  - Class weight computation for imbalance handling in models

WHAT COMES IN / WHAT GOES OUT
-------------------------------
  Input  : phase1_features.parquet  (18 columns, ~19k rows for 500pts)
  Output : phase1_train.parquet
           phase1_test.parquet
           preprocessor.joblib      (fitted scaler, for inference later)

REMAINING MISSINGNESS AFTER features.py
-----------------------------------------
  From the 500-patient test run:
    Temp          : ~64.8%  (highest — no algebraic recovery possible)
    DBP           : ~16.2%  (residual after algebraic recovery)
    ShockIndex    : ~15.8%  (inherits SBP missingness)
    PulsePressure : ~16.2%  (inherits DBP missingness)
    SBP           : ~15.6%
    O2Sat         : ~12.8%
    MAP           : ~10.6%
    Resp          : ~11.7%
    HR            :  ~8.3%
    Age, Gender, ICULOS, AgeGroup : 0% (complete)

THREE-TIER IMPUTATION STRATEGY
--------------------------------
  Tier 1 — Forward-fill within patient (ffill)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  For each patient's time series, carry the last known value forward
  across subsequent hours. Clinically justified: a vital reading
  remains approximately valid for a short period after measurement,
  especially in a monitored ICU setting.
  Applied per-patient group (groupby PatientID), sorted by ICULOS.

  Tier 2 — Backward-fill within patient (bfill)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  For rows at the START of a patient's stay (before any reading exists),
  carry the first available reading backward. Handles the case where a
  patient's first few ICU hours have no measurement yet.
  Applied after Tier 1.

  Tier 3 — Global column median (fallback)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  For patients who have ZERO readings for a vital across their entire
  ICU stay (e.g., Temp was never recorded for that patient at all),
  forward/backward fill within-patient has nothing to propagate.
  These residual NaNs are filled with the column median computed from
  the TRAINING set only (to prevent data leakage from test set).

  TRIPOD compliance note (Item 9):
  This three-tier strategy is documented here per TRIPOD requirements.
  The global median is fit ONLY on training data and applied to test,
  avoiding target leakage. This is enforced by the pipeline structure.

FEATURES USED FOR MODEL TRAINING
----------------------------------
  Continuous (scaled):
    HR, O2Sat, Temp, SBP, MAP, DBP, Resp,
    ShockIndex, PulsePressure, NEWS2Score

  Ordinal / integer (not scaled — already on meaningful numeric scales):
    Age, AgeGroup, Gender, ICULOS

  Dropped before training:
    PatientID    — identifier, not a feature
    SepsisLabel  — target signal already consumed by labeling.py
    UrgencyLabel — string version of target (UrgencyLevel is used)

  Target:
    UrgencyLevel — integer 0/1/2/3

TRAIN / TEST SPLIT
-------------------
  80% train / 20% test, stratified by UrgencyLevel.
  Random seed fixed at 42 for reproducibility (TRIPOD Item 10b).
  Split is done at the ROW level (patient-hour level), not patient level.

  NOTE on patient-level leakage:
  A stricter split would separate entire patients into train vs test
  (so the model never sees any hours from a test patient during training).
  This is the gold standard for clinical ML. For Phase 1 of this project,
  we use row-level splitting for simplicity and note this as a limitation.
  A patient-level split is marked as a Phase 2 improvement item.

CLASS WEIGHTS
--------------
  Computed from training set label distribution and stored in the output
  for direct use in sklearn model constructors (class_weight parameter)
  and XGBoost (sample_weight).

  Formula: weight_c = n_total / (n_classes * n_samples_in_class_c)
  This is sklearn's 'balanced' class weight formula, computed explicitly
  so it can be applied consistently across all model families.

Usage
-----
    from src.preprocessing import run_preprocessing_pipeline

    train_df, test_df, meta = run_preprocessing_pipeline(
        "data/processed/phase1/phase1_features.parquet"
    )

Or run directly:
    python src/preprocessing.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ======================================================================
# COLUMN DEFINITIONS
# ======================================================================

# Continuous vitals/indices — will be imputed then StandardScaled
CONTINUOUS_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "ShockIndex", "PulsePressure", "NEWS2Score",
]

# Ordinal/integer features — imputed if needed, NOT scaled
ORDINAL_FEATURES = [
    "Age", "AgeGroup", "Gender", "ICULOS",
]

# All model input features (order preserved for downstream use)
ALL_FEATURES = CONTINUOUS_FEATURES + ORDINAL_FEATURES

# Target column
TARGET = "UrgencyLevel"

# Columns to drop before training (identifiers / consumed signals)
DROP_COLS = ["PatientID", "SepsisLabel", "UrgencyLabel"]


# ======================================================================
# TIER 1 + 2: WITHIN-PATIENT FORWARD/BACKWARD FILL
# ======================================================================

def _within_patient_fill(df: pd.DataFrame, vital_cols: list[str]) -> pd.DataFrame:
    """
    Apply forward-fill then backward-fill within each patient's time
    series, sorted by ICULOS (ICU hour index).

    This is the most clinically defensible imputation for time-series
    vital signs: a measured value stays valid until the next reading.

    Parameters
    ----------
    df         : DataFrame with PatientID, ICULOS, and vital columns
    vital_cols : list of column names to impute

    Returns
    -------
    DataFrame with within-patient NaNs filled where possible.
    """
    df = df.sort_values(["PatientID", "ICULOS"]).copy()

    df[vital_cols] = (
        df.groupby("PatientID")[vital_cols]
        .transform(lambda x: x.ffill().bfill())
    )

    return df


# ======================================================================
# TIER 3: GLOBAL MEDIAN FALLBACK (fit on train only)
# ======================================================================

def _fit_median_imputer(train_df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """
    Compute column medians from training data only.
    Returns a dict {column_name: median_value} for later application.
    """
    medians = {}
    for col in cols:
        median_val = train_df[col].median()
        medians[col] = median_val
        logger.info(f"  Median fallback — {col}: {median_val:.4f}")
    return medians


def _apply_median_imputer(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Apply pre-computed medians to fill residual NaNs."""
    for col, val in medians.items():
        before = df[col].isna().sum()
        df[col] = df[col].fillna(val)
        after = df[col].isna().sum()
        if before > 0:
            logger.info(f"  Median fill — {col}: {before} NaNs -> {after} NaNs")
    return df


# ======================================================================
# SCALING
# ======================================================================

def _fit_scaler(train_df: pd.DataFrame, cols: list[str]) -> StandardScaler:
    """Fit StandardScaler on training continuous features."""
    scaler = StandardScaler()
    scaler.fit(train_df[cols])
    return scaler


def _apply_scaler(df: pd.DataFrame, scaler: StandardScaler,
                  cols: list[str]) -> pd.DataFrame:
    """Apply fitted scaler to a DataFrame split."""
    df = df.copy()
    df[cols] = scaler.transform(df[cols])
    return df


# ======================================================================
# CLASS WEIGHT COMPUTATION
# ======================================================================

def compute_class_weights(y_train: pd.Series) -> dict[int, float]:
    """
    Compute balanced class weights from training labels.

    Formula (sklearn 'balanced'):
        weight_c = n_total / (n_classes * n_samples_in_class_c)

    Returns dict {class_int: weight_float} for use in:
      - sklearn models: class_weight=weights_dict
      - XGBoost:        scale_pos_weight (binary) or sample_weight array
      - All models in models.py

    Parameters
    ----------
    y_train : pd.Series of UrgencyLevel integers (0, 1, 2, 3)

    Returns
    -------
    dict mapping each class integer to its float weight
    """
    n_total   = len(y_train)
    n_classes = y_train.nunique()
    weights   = {}

    for cls in sorted(y_train.unique()):
        n_cls        = (y_train == cls).sum()
        weights[cls] = n_total / (n_classes * n_cls)

    return weights


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_preprocessing_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "data/processed/phase1",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline end to end.

    Steps:
      1. Load phase1_features.parquet
      2. Drop non-feature columns
      3. Within-patient forward/backward fill (Tiers 1+2)
      4. Stratified train/test split (80/20)
      5. Global median imputation fit on train, applied to both (Tier 3)
      6. StandardScaler fit on train, applied to both
      7. Compute class weights from training labels
      8. Save train/test parquets + fitted preprocessor artifacts

    Parameters
    ----------
    input_path   : path to phase1_features.parquet
    output_dir   : directory to save outputs
    test_size    : fraction of rows for test set (default 0.20)
    random_state : random seed for reproducibility (default 42)

    Returns
    -------
    train_df : model-ready training DataFrame
    test_df  : model-ready test DataFrame
    meta     : dict with keys:
                 'class_weights'  : dict {int: float}
                 'feature_cols'   : list of feature column names
                 'target_col'     : str
                 'scaler'         : fitted StandardScaler
                 'medians'        : dict {col: float}
                 'train_rows'     : int
                 'test_rows'      : int
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    logger.info(f"  {len(df):,} rows, {len(df.columns)} columns loaded.")

    # ------------------------------------------------------------------
    # 2. Tier 1+2: Within-patient fill BEFORE splitting
    #    (uses only that patient's own data — no leakage risk)
    # ------------------------------------------------------------------
    logger.info("Applying within-patient forward/backward fill...")
    fill_cols = [c for c in CONTINUOUS_FEATURES if c in df.columns]
    df = _within_patient_fill(df, fill_cols)

    remaining_missing = df[fill_cols].isna().sum()
    logger.info(f"Remaining NaNs after within-patient fill:\n{remaining_missing[remaining_missing > 0]}")

    # ------------------------------------------------------------------
    # 3. Prepare feature matrix and target
    # ------------------------------------------------------------------
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    extra_cols = [c for c in [TARGET, "PatientID", "ICULOS"] if c not in feature_cols]
    X = df[feature_cols + extra_cols].copy()

    # ------------------------------------------------------------------
    # 4. Stratified train/test split
    # ------------------------------------------------------------------
    logger.info(f"Splitting: {1-test_size:.0%} train / {test_size:.0%} test "
                f"(stratified by {TARGET}, seed={random_state})...")

    train_df, test_df = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
        stratify=X[TARGET],
    )

    logger.info(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    # ------------------------------------------------------------------
    # 5. Tier 3: Global median — fit on TRAIN only, apply to both
    # ------------------------------------------------------------------
    logger.info("Computing global median fallback (train only)...")
    residual_cols = [c for c in fill_cols if train_df[c].isna().any()]
    medians = _fit_median_imputer(train_df, residual_cols)

    train_df = _apply_median_imputer(train_df.copy(), medians)
    test_df  = _apply_median_imputer(test_df.copy(), medians)

    # ------------------------------------------------------------------
    # 6. StandardScaler — fit on TRAIN only, apply to both
    # ------------------------------------------------------------------
    logger.info("Fitting StandardScaler on training continuous features...")
    scale_cols = [c for c in CONTINUOUS_FEATURES if c in feature_cols]
    scaler = _fit_scaler(train_df, scale_cols)

    train_df = _apply_scaler(train_df, scaler, scale_cols)
    test_df  = _apply_scaler(test_df,  scaler, scale_cols)

    # ------------------------------------------------------------------
    # 7. Class weights
    # ------------------------------------------------------------------
    class_weights = compute_class_weights(train_df[TARGET])
    logger.info(f"Class weights: {class_weights}")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    train_path = output_dir / "phase1_train.parquet"
    test_path  = output_dir / "phase1_test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path,   index=False)
    logger.info(f"Saved train -> {train_path}")
    logger.info(f"Saved test  -> {test_path}")

    # Save preprocessor artifacts for reproducibility / inference
    preprocessor = {"scaler": scaler, "medians": medians}
    preprocessor_path = output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Saved preprocessor -> {preprocessor_path}")

    meta = {
        "class_weights" : class_weights,
        "feature_cols"  : feature_cols,
        "target_col"    : TARGET,
        "scaler"        : scaler,
        "medians"       : medians,
        "train_rows"    : len(train_df),
        "test_rows"     : len(test_df),
    }

    return train_df, test_df, meta


# ======================================================================
# SUMMARY
# ======================================================================

def summarize_preprocessing(train_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             meta: dict) -> None:
    """Print a validation summary of the preprocessing output."""
    print("\n=== Preprocessing Summary ===")

    label_names = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

    for split_name, split_df in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n  {split_name} SET ({len(split_df):,} rows):")
        counts = split_df[TARGET].value_counts().sort_index()
        for lvl, cnt in counts.items():
            print(f"    {label_names[lvl]:<10} {cnt:>6,}  ({cnt/len(split_df)*100:.1f}%)")

    print(f"\n  Class weights (for model training):")
    for cls, weight in meta["class_weights"].items():
        print(f"    Class {cls} ({label_names[cls]:<10}) : {weight:.4f}")

    print(f"\n  Feature columns ({len(meta['feature_cols'])}):")
    print(f"    {meta['feature_cols']}")

    print(f"\n  Remaining NaNs after full imputation:")
    feature_cols = meta["feature_cols"]
    train_nans = train_df[feature_cols].isna().sum().sum()
    test_nans  = test_df[feature_cols].isna().sum().sum()
    print(f"    Train : {train_nans}")
    print(f"    Test  : {test_nans}")
    if train_nans == 0 and test_nans == 0:
        print("    [OK] No NaNs remain — data is model-ready.")
    else:
        print("    [WARNING] NaNs remain — investigate before training.")


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":
    train_df, test_df, meta = run_preprocessing_pipeline(
        input_path="data/processed/phase1/phase1_features.parquet",
        output_dir="data/processed/phase1",
    )
    summarize_preprocessing(train_df, test_df, meta)