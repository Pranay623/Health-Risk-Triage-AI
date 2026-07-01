# Health Risk Triage — Phase 1 Model Comparison Results
## Overview
Dataset: PhysioNet 2019 Sepsis Challenge (500-patient test slice)
Task: 4-class urgency classification (Low / Medium / High / Critical)
Label derivation: NEWS2-style vital scoring + SepsisLabel override
Features: 14 (vitals + engineered: ShockIndex, PulsePressure, AgeGroup, NEWS2Score)
Train/Test split: 80/20 stratified by UrgencyLevel
Class imbalance: handled via class_weight='balanced'

## Summary Table
| Model | F1-Macro | F1-Weighted | AUROC | Critical Recall | Critical Precision |
|-------|----------|-------------|-------|-----------------|--------------------|
| xgboost | 0.9423 | 0.9914 | 0.9983 | 0.798 | 0.8587 |
| random_forest | 0.851 | 0.9749 | 0.9807 | 0.4747 | 0.5165 |
| decision_tree | 0.803 | 0.9511 | 0.962 | 0.697 | 0.2413 |
| mlp | 0.7711 | 0.9675 | 0.9675 | 0.1515 | 0.5556 |
| logistic_regression | 0.5596 | 0.7443 | 0.7979 | 0.1313 | 0.014 |

## Per-Model Details
### xgboost
- **F1-Macro**: 0.9423
- **AUROC**: 0.9983

**Per-Class Metrics:**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Low | 0.9973 | 0.9973 | 0.9973 | 3376 |
| Medium | 0.985 | 0.991 | 0.988 | 332 |
| High | 0.9277 | 0.9872 | 0.9565 | 78 |
| Critical | 0.8587 | 0.798 | 0.8272 | 99 |

### random_forest
- **F1-Macro**: 0.851
- **AUROC**: 0.9807

**Per-Class Metrics:**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Low | 0.9899 | 0.9876 | 0.9887 | 3376 |
| Medium | 0.9792 | 0.994 | 0.9865 | 332 |
| High | 0.8764 | 1.0 | 0.9341 | 78 |
| Critical | 0.5165 | 0.4747 | 0.4947 | 99 |

### decision_tree
- **F1-Macro**: 0.803
- **AUROC**: 0.962

**Per-Class Metrics:**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Low | 0.9959 | 0.9434 | 0.969 | 3376 |
| Medium | 0.9808 | 0.9217 | 0.9503 | 332 |
| High | 0.8764 | 1.0 | 0.9341 | 78 |
| Critical | 0.2413 | 0.697 | 0.3584 | 99 |

### mlp
- **F1-Macro**: 0.7711
- **AUROC**: 0.9675

**Per-Class Metrics:**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Low | 0.982 | 0.9994 | 0.9906 | 3376 |
| Medium | 0.9698 | 0.9669 | 0.9683 | 332 |
| High | 0.8242 | 0.9615 | 0.8876 | 78 |
| Critical | 0.5556 | 0.1515 | 0.2381 | 99 |

### logistic_regression
- **F1-Macro**: 0.5596
- **AUROC**: 0.7979

**Per-Class Metrics:**
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Low | 0.9802 | 0.6448 | 0.7779 | 3376 |
| Medium | 0.4538 | 0.8584 | 0.5938 | 332 |
| High | 0.7333 | 0.9872 | 0.8415 | 78 |
| Critical | 0.014 | 0.1313 | 0.0252 | 99 |

## SHAP Feature Importance (Best Model)
Model: **xgboost**

| Rank | Feature | Mean |SHAP| |
|------|---------|--------|
| 1 | NEWS2Score | 3.5634 |
| 2 | Age | 0.7152 |
| 3 | ICULOS | 0.6442 |
| 4 | Resp | 0.3146 |
| 5 | Temp | 0.3099 |
| 6 | ShockIndex | 0.2808 |
| 7 | HR | 0.2673 |
| 8 | DBP | 0.1857 |
| 9 | SBP | 0.1794 |
| 10 | O2Sat | 0.1708 |
| 11 | PulsePressure | 0.1489 |
| 12 | MAP | 0.1353 |
| 13 | Gender | 0.1090 |
| 14 | AgeGroup | 0.0247 |

## Notes
- SVM excluded: RBF kernel + Platt scaling ran >2.5h on 15k rows (O(n^2) complexity).
- Row-level train/test split used (not patient-level). Patient-level split is a Phase 2 improvement.
- Labels derived from NEWS2 thresholds + SepsisLabel override. See labeling.py for full documentation.
- TRIPOD+AI reporting checklist applied throughout. See individual src/ files for Item references.
