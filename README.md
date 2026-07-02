# Health Risk Triage AI

An interpretable ML system for early health risk screening and urgency triage using vitals-based physiological inputs. Benchmarks 5 model families (logistic regression → gradient boosting → neural nets) on PhysioNet 2019 sepsis data, with SHAP-based explainability and TRIPOD+AI-aligned transparency reporting.

> **Project context:** Built during a summer research internship at DRDO DMSRDE, Kanpur. Grounded in WHO ETAT/IMCI triage principles and the clinical EWS literature.

---

## The Problem

In low-resource or first-contact clinical settings, patients deteriorate before anyone notices. Traditional Early Warning Systems (NEWS, MEWS) use rigid point-based rules that treat vitals as independent snapshots — they miss the non-linear interactions between age, trajectory, and vital-sign combinations that actually predict deterioration.

This project asks: **can a model trained only on easily obtainable vitals classify urgency level well enough to support real triage decisions?**

---

## What It Does

- Parses 40,336 ICU patient records (PhysioNet 2019 Sepsis Challenge)
- Derives a 4-level urgency scale (Low / Medium / High / Critical) using NEWS2-style vital scoring + SepsisLabel ground truth
- Engineers clinical composite features: Shock Index, Pulse Pressure, Age Group
- Trains and compares 5 ML model families with stratified CV and class-imbalance handling
- Explains predictions using per-class SHAP values — identifying which vitals drive each urgency level

---

## Key Results (500-patient test slice)

| Model | F1-Macro | AUROC | Critical Recall | Critical Precision |
|---|---|---|---|---|
| **XGBoost** | **0.9423** | **0.9983** | **0.7980** | **0.8587** |
| Random Forest | 0.8510 | 0.9807 | 0.4747 | 0.5165 |
| Decision Tree | 0.8030 | 0.9620 | 0.6970 | 0.2413 |
| MLP | 0.7711 | 0.9675 | 0.1515 | 0.5556 |
| Logistic Regression | 0.5596 | 0.7979 | 0.1313 | 0.0140 |

> SVM excluded: RBF kernel + Platt scaling ran >2.5 hours on 15k training rows — computationally infeasible at this scale, consistent with known O(n²) complexity limitations documented in the EWS literature.

### SHAP Finding — Why ML beats hardcoded rules here

For **Low, Medium, and High** urgency patients, the model relies heavily on `NEWS2Score` (the aggregate vital severity score) — exactly as clinical rules would.

For **Critical** patients, `NEWS2Score` drops to position 5. The top drivers become `Age`, `ICULOS` (ICU length of stay), `Temp`, and `HR`. The model learned that sepsis-driven critical deterioration is **context-dependent** — a pure NEWS2 cutoff would systematically miss these cases.

---

## Architecture

```
health-risk-triage/
│
├── data/
│   ├── raw/physionet2019/        # 40,336 PSV files (not committed — see below)
│   └── processed/phase1/         # parquet outputs at each pipeline stage
│
├── src/
│   ├── data_loading.py           # PSV → unified Phase 1 DataFrame
│   ├── labeling.py               # NEWS2 scoring + SepsisLabel → urgency labels
│   ├── features.py               # Shock Index, Pulse Pressure, AgeGroup, MAP recovery
│   ├── preprocessing.py          # 3-tier imputation, scaling, stratified split
│   ├── models.py                 # 5 model family definitions + hyperparameter grids
│   ├── train.py                  # RandomizedSearchCV + 5-fold CV + model fitting
│   ├── evaluate.py               # Test-set metrics: F1, AUROC, per-class P/R
│   └── explain.py                # Per-class SHAP breakdown
│
├── models/                       # Saved .joblib model artifacts
├── reports/
│   ├── figures/                  # SHAP rankings per model
│   ├── cv_results.json           # Cross-validation scores + best hyperparameters
│   └── results_summary.md        # Full model comparison report
│
├── notebooks/                    # Jupyter notebooks (exploration + analysis)
├── config.yaml                   # All thresholds, paths, seeds in one place
└── requirements.txt              # Pinned dependency stack
```

---

## Pipeline

Each `src/` file is a self-contained stage. Run them in order:

```bash
# 1. Parse raw PSV files → Phase 1 DataFrame
python -m src.data_loading

# 2. Derive urgency labels (NEWS2 + SepsisLabel)
python -m src.labeling

# 3. Engineer clinical features
python -m src.features

# 4. Impute, scale, split
python -m src.preprocessing

# 5. Train all 5 models with CV
python -m src.train

# 6. Evaluate on held-out test set
python -m src.evaluate

# 7. Per-class SHAP explainability
python -m src.explain
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Pranay623/Health-Risk-Triage-AI
cd health-risk-triage-ai
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Get the data**

The raw PhysioNet 2019 data is not committed to this repo (40,336 PSV files, ~1GB). Download it:

- Register at [physionet.org](https://physionet.org/content/challenge-2019/1.0.0/) and accept the data use agreement
- Download `training_setA.zip` and `training_setB.zip`
- Extract both into `data/raw/physionet2019/`

Expected structure after extraction:
```
data/raw/physionet2019/
├── training_setA/
│   ├── p000001.psv
│   ├── p000002.psv
│   └── ...  (~20,000 files)
└── training_setB/
    ├── p100000.psv
    └── ...  (~20,000 files)
```

**4. Run the pipeline**
```bash
python -m src.data_loading
# ... follow pipeline order above
```

---

## Labeling Methodology

Raw PhysioNet data provides only a binary `SepsisLabel` (0/1) per hour. This project derives a clinically grounded **4-level urgency scale** using two independent signals:

**Signal A — NEWS2-style vital severity scoring**
Five vitals (HR, Resp, O2Sat, SBP, Temp) are each scored 0–3 points based on deviation from normal ranges per RCP NEWS2 2017 thresholds. Scores sum to a total (0–15), mapped to urgency bands:

| Score | Urgency | Clinical response |
|---|---|---|
| 0–4 | Low | Routine monitoring |
| 5–6 | Medium | Urgent review |
| 7–8 | High | Emergency assessment |
| 9+ | Critical | Immediate stabilisation |

**Signal B — SepsisLabel override**
Any hour with `SepsisLabel=1` is forced to Critical regardless of NEWS2 score. Clinically justified: sepsis onset (Sepsis-3 criteria) is itself a critical escalation trigger in all major triage frameworks.

This two-signal design means Critical is not just "SepsisLabel renamed" — 62 rows (12.5% of Critical) were classified Critical by vital severity alone, independent of the sepsis flag.

---

## Clinical Framework

This project is not a diagnostic system. It is a **triage support tool** — it estimates acuity and likely intervention need from easily obtainable vitals, consistent with WHO ETAT principles:

> *"In a crisis, you don't diagnose; you stabilize."*

Clinical references:
- **NEWS2**: Royal College of Physicians, 2017
- **WHO ETAT**: Emergency Triage Assessment and Treatment, WHO 2005
- **IMCI**: Integrated Management of Childhood Illness, WHO 2019
- **TRIPOD+AI**: Collins et al., Ann Intern Med 2015 + AI extension 2024
- **EWS ML review**: Gerry et al., JMIR 2021;23(2):e25187

---

## Transparency (TRIPOD+AI)

This project follows TRIPOD+AI reporting guidelines throughout:

| Item | Where documented |
|---|---|
| Data source | `data_loading.py` docstring + this README |
| Missing data handling | `preprocessing.py` docstring (3-tier strategy) |
| Label derivation | `labeling.py` docstring (NEWS2 thresholds + SepsisLabel override) |
| Feature engineering | `features.py` docstring (algebraic derivations + clinical indices) |
| Train/test split | `preprocessing.py` (80/20 stratified, seed=42) |
| Hyperparameters | `models.py` + `reports/cv_results.json` |
| Evaluation metrics | `evaluate.py` (F1-macro primary, AUROC secondary, per-class P/R/F1) |
| Explainability | `explain.py` (per-class SHAP, TreeExplainer for XGBoost) |
| Known limitations | See below |

---

## Known Limitations

- **Row-level train/test split**: rows from the same patient may appear in both train and test. A patient-level split is more rigorous and is planned for Phase 2.
- **500-patient test slice**: full results will be validated on all 40,336 patients.
- **SVM excluded**: computational infeasibility documented, not a modelling gap.
- **No consciousness level (AVPU/GCS)**: not available in PhysioNet 2019. Planned for Phase 2 with symptom flags.
- **Missing vital imputation**: Temp has ~65% missingness; within-patient fill + median fallback used. See `preprocessing.py`.
- **NEWS2 Scale 1 used for O2Sat**: Scale 2 (for hypercapnic patients) not applicable without diagnosis flags.

---

## Phase 2 Roadmap

- [ ] Full 40,336-patient run and final benchmark
- [ ] Patient-level train/test split
- [ ] Add symptom flags: chest pain, cough, breathlessness, AVPU/GCS
- [ ] Syndrome-group classification (respiratory / circulatory / neurologic / infectious)
- [ ] Synthetic data augmentation for rare critical cases
- [ ] LSTM/GRU time-series model (using ICULOS as temporal axis)
- [ ] Interactive triage dashboard (React + FastAPI inference endpoint)

---

## Author

**Pranay** — B.Tech CSE (AI & ML), AKGEC Ghaziabad  
Built during summer research internship at DRDO DMSRDE, Kanpur