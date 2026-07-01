"""
models.py
==========
Model definitions and hyperparameter grids for the Phase 1
multi-model comparison in the Health Risk Triage project.

SIX MODEL FAMILIES
-------------------
Per the project specification and literature review, we benchmark
one representative from each major ML family used in clinical EWS
research (based on the scoping review: JMIR 2021;23(2):e25187):

  Family            | Model                  | sklearn class
  ------------------|------------------------|---------------------------
  Linear/Statistical| Logistic Regression    | LogisticRegression
  Tree              | Decision Tree          | DecisionTreeClassifier
  Ensemble (bagging)| Random Forest          | RandomForestClassifier
  Ensemble (boost)  | XGBoost                | XGBClassifier
  Kernel            | Support Vector Machine | SVC
  Neural            | MLP (Multilayer        | MLPClassifier
                    |   Perceptron)          |

WHY THESE SIX SPECIFICALLY
----------------------------
  Logistic Regression : Clinically acceptable baseline; maximally
    interpretable; coefficients map directly to feature importance.
    Used as the floor — if LR is competitive, simpler is better.

  Decision Tree : Produces explicit IF-THEN rule paths that mirror
    ETAT/NEWS2 clinical logic. A shallow tree (max_depth 4-6) can
    be printed and read by a clinician. Valuable for explainability
    even if accuracy is lower than ensemble methods.

  Random Forest : Strong tabular baseline per literature. Handles
    non-linearity and feature interactions without feature engineering.
    Tends to outperform LR on clinical tabular data with correlated
    features (e.g. SBP, MAP, DBP are algebraically related).

  XGBoost : Typically best-performing on structured medical tabular
    data in head-to-head comparisons (literature: 8/9 ML studies
    outperformed traditional EWS). Gradient boosting on residuals
    captures subtle deterioration patterns. Supports sample_weight
    for class imbalance handling.

  SVM (RBF kernel) : Useful comparison for medium-complexity decision
    boundaries. Less common in recent clinical ML but provides a
    non-parametric kernel baseline. Computationally heavier on large
    datasets — will be tested on the 500-patient slice first.

  MLP : Neural baseline for tabular classification. 2-3 hidden layers.
    Not expected to dramatically outperform XGBoost on this dataset
    size, but its inclusion allows the comparison to cover the neural
    family as stated in the project specification.

CLASS IMBALANCE HANDLING
--------------------------
  All models receive class_weight or sample_weight information.
  Class weights are computed in preprocessing.py and passed in at
  training time. See train.py for the application pattern.

  sklearn models: class_weight='balanced' parameter (computes
    automatically from y_train, equivalent to our precomputed weights)
  XGBoost: sample_weight array passed to fit() call

HYPERPARAMETER GRIDS
----------------------
  Each model has an associated hyperparameter grid for use with
  sklearn's GridSearchCV or RandomizedSearchCV in train.py.
  Grids are intentionally moderate in size — enough to tune the
  most impactful parameters without excessive compute time.

  The most impactful parameters per model family:
    LR    : C (regularization strength), solver
    DT    : max_depth, min_samples_leaf
    RF    : n_estimators, max_depth, min_samples_leaf
    XGB   : n_estimators, max_depth, learning_rate, subsample
    SVM   : C, gamma
    MLP   : hidden_layer_sizes, alpha (L2 reg), learning_rate_init

REPRODUCIBILITY
----------------
  All models accept a random_state parameter fixed at RANDOM_STATE=42.
  This is consistent with the train/test split seed in preprocessing.py.
  Per TRIPOD+AI Item 15b, all hyperparameters and random seeds are
  documented here for full reproducibility.

Usage
-----
    from src.models import get_all_models, get_all_param_grids

    models     = get_all_models()
    param_grids = get_all_param_grids()

    # models['xgboost'] → fitted XGBClassifier instance
    # param_grids['xgboost'] → dict of hyperparameter search space
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. Run: pip install xgboost")
    print("          XGBoost model will be skipped in comparisons.")


# ======================================================================
# GLOBAL CONSTANTS
# ======================================================================

RANDOM_STATE = 42      # Fixed seed — matches preprocessing.py split seed
N_CLASSES    = 4       # Low, Medium, High, Critical


# ======================================================================
# MODEL DEFINITIONS
# ======================================================================

def get_logistic_regression(class_weight: str | dict = "balanced") -> LogisticRegression:
    """
    Logistic Regression — Linear/Statistical baseline.

    solver='saga' supports L1 and L2 penalties and scales well to
    larger datasets. max_iter=2000 handles convergence on our 14-feature
    scaled input. multi_class='multinomial' is correct for 4-class
    ordinal prediction (vs 'ovr' one-vs-rest which is less appropriate
    for ordered classes).

    class_weight='balanced' lets sklearn compute weights from y_train
    automatically — equivalent to our precomputed weights dict.
    """
    return LogisticRegression(
        C=1.0,
        solver="saga",
        penalty="l2",
        multi_class="multinomial",
        max_iter=2000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_decision_tree(class_weight: str | dict = "balanced") -> DecisionTreeClassifier:
    """
    Decision Tree — interpretable rule-path baseline.

    max_depth=6 produces a tree readable by a clinician (64 max leaves).
    Shallower than unconstrained trees to prioritise interpretability
    over raw accuracy. min_samples_leaf=20 prevents overfitting to
    rare Critical rows in a small training set.

    criterion='gini' is standard; 'entropy' will be tested in the grid.
    """
    return DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=20,
        criterion="gini",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def get_random_forest(class_weight: str | dict = "balanced") -> RandomForestClassifier:
    """
    Random Forest — ensemble bagging baseline.

    n_estimators=200 is a reasonable starting point; more trees improve
    stability but with diminishing returns. max_features='sqrt' is the
    standard for classification (sqrt of 14 features = ~3-4 per split).
    n_jobs=-1 uses all available cores for parallel tree building.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_xgboost() -> "XGBClassifier":
    """
    XGBoost — gradient boosting ensemble (typically best on tabular data).

    num_class=4 required for multi-class with objective='multi:softprob'.
    eval_metric='mlogloss' is appropriate for multi-class probability
    calibration. scale_pos_weight is NOT used here (it's binary-only);
    class imbalance is handled via sample_weight in train.py instead.

    use_label_encoder=False suppresses a deprecation warning in older
    XGBoost versions.
    tree_method='hist' is faster than 'exact' for larger datasets.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")

    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def get_svm() -> SVC:
    """
    Support Vector Machine — kernel method baseline.

    kernel='rbf' is standard for non-linear tabular classification.
    probability=True required to produce predict_proba output (needed
    for AUROC computation in evaluate.py). This adds compute overhead
    via Platt scaling — acceptable for our dataset size.

    class_weight='balanced' handles imbalance.
    C=1.0 and gamma='scale' are sklearn defaults — tuned in grid search.

    NOTE: SVM is the most computationally expensive model here.
    On the 500-patient slice (~15k train rows) it should complete
    in under a minute. On the full 40k dataset it may take 10-30 min.
    """
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        decision_function_shape="ovr",
    )


def get_mlp() -> MLPClassifier:
    """
    Multilayer Perceptron — neural network baseline.

    Architecture: two hidden layers (64, 32 neurons).
    Small enough to train quickly on tabular data of this size.
    Larger architectures (128, 64, 32) will be tested in grid search.

    alpha=0.001 is L2 regularization (weight decay) — important for
    preventing overfitting on the minority High/Critical classes.
    early_stopping=True uses 10% of training data as a validation set
    and stops if validation loss doesn't improve for n_iter_no_change
    consecutive epochs. This prevents overfitting without manual epoch
    tuning and is especially important given class imbalance.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )


# ======================================================================
# HYPERPARAMETER GRIDS
# ======================================================================

def get_all_param_grids() -> dict[str, dict]:
    """
    Hyperparameter search grids for GridSearchCV / RandomizedSearchCV.

    Grids are sized for RandomizedSearchCV (n_iter=20 in train.py)
    rather than exhaustive GridSearchCV, to keep compute tractable.

    Each grid contains the 2-3 most impactful parameters per model,
    per standard ML tuning guidance for each algorithm family.
    """
    grids = {
        "logistic_regression": {
            "C"      : [0.01, 0.1, 1.0, 10.0, 100.0],
            "penalty": ["l1", "l2"],
            "solver" : ["saga"],            # saga supports both l1 and l2
        },

        "decision_tree": {
            "max_depth"       : [3, 4, 5, 6, 8, 10, None],
            "min_samples_leaf": [5, 10, 20, 50],
            "criterion"       : ["gini", "entropy"],
        },

        "random_forest": {
            "n_estimators"    : [100, 200, 300],
            "max_depth"       : [5, 10, 20, None],
            "min_samples_leaf": [1, 5, 10],
            "max_features"    : ["sqrt", "log2"],
        },

        "xgboost": {
            "n_estimators"   : [100, 200, 300],
            "max_depth"      : [3, 5, 6, 8],
            "learning_rate"  : [0.01, 0.05, 0.1, 0.2],
            "subsample"      : [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        },

        "svm": {
            "C"    : [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.001, 0.01],
            "kernel": ["rbf", "poly"],
        },

        "mlp": {
            "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32), (64,)],
            "alpha"             : [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
        },
    }

    return grids


# ======================================================================
# UNIFIED MODEL REGISTRY
# ======================================================================

def get_all_models() -> dict[str, object]:
    """
    Return all 6 model instances in a single dict.

    Keys match the keys in get_all_param_grids() for consistent
    lookup in train.py and evaluate.py.

    Returns
    -------
    dict mapping model_name (str) -> unfitted sklearn-compatible estimator
    """
    models = {
        "logistic_regression": get_logistic_regression(),
        "decision_tree"      : get_decision_tree(),
        "random_forest"      : get_random_forest(),
        # SVM EXCLUDED — COMPUTATIONAL FEASIBILITY
        # -----------------------------------------
        # SVM with RBF kernel + Platt scaling (probability=True) requires
        # O(n^2)-O(n^3) compute. On the 15k-row training slice, RandomizedSearchCV
        # (20 iter x 5 folds = 100 fits) ran for >2.5 hours without completing.
        # This confirms findings in the EWS literature that kernel SVMs are
        # impractical for real-time or large-scale clinical deployment.
        # Excluded from Phase 1 comparison. Noted as a documented limitation.
        # Reference: Computational complexity of SVMs — Bottou & Lin, 2007.
        #
        # "svm"                : get_svm(),
        "mlp"                : get_mlp(),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = get_xgboost()

    return models


# ======================================================================
# QUICK SMOKE TEST
# ======================================================================

if __name__ == "__main__":
    print("=== Model Registry Smoke Test ===\n")

    models      = get_all_models()
    param_grids = get_all_param_grids()

    for name, model in models.items():
        grid        = param_grids.get(name, {})
        param_count = sum(len(v) for v in grid.values())
        print(f"  {name:<25} {type(model).__name__:<30} grid params: {param_count}")

    print(f"\n  Total models registered: {len(models)}")
    print("  [OK] All models instantiated successfully.")