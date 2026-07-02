"""
src/__init__.py
================
Makes `src` an importable Python package.

Exposes the clean public API for the Health Risk Triage pipeline
so notebooks and scripts can import with short paths:

    from src import load_physionet_phase1
    from src import assign_urgency_labels
    from src import engineer_features
    from src import run_preprocessing_pipeline
    from src import get_all_models, get_all_param_grids
    from src import run_training_pipeline
    from src import run_evaluation_pipeline
    from src import run_explain_pipeline

Pipeline execution order:
    1. data_loading   -> load_physionet_phase1()
    2. labeling       -> assign_urgency_labels()
    3. features       -> engineer_features()
    4. preprocessing  -> run_preprocessing_pipeline()
    5. models         -> get_all_models()
    6. train          -> run_training_pipeline()
    7. evaluate       -> run_evaluation_pipeline()
    8. explain        -> run_explainability()
"""

from src.data_loading  import load_physionet_phase1
from src.labeling      import assign_urgency_labels, summarize_labels
from src.features      import engineer_features, summarize_features
from src.preprocessing import run_preprocessing_pipeline, compute_class_weights
from src.models        import get_all_models, get_all_param_grids
from src.train         import run_training_pipeline
from src.evaluate      import run_evaluation_pipeline
from src.explain       import run_explainability

__all__ = [
    "load_physionet_phase1",
    "assign_urgency_labels",
    "summarize_labels",
    "engineer_features",
    "summarize_features",
    "run_preprocessing_pipeline",
    "compute_class_weights",
    "get_all_models",
    "get_all_param_grids",
    "run_training_pipeline",
    "run_evaluation_pipeline",
    "run_explainability",
]