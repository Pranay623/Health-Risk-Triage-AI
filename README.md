# Health Risk Triage AI

An interpretable ML system and interactive dashboard for early health risk screening and urgency triage using vitals-based physiological inputs. Benchmarks 5 model families on PhysioNet 2019 sepsis data, with SHAP-based explainability and a real-time Streamlit web interface.

> **Project context:** Built during a summer research internship at DRDO DMSRDE, Kanpur. Grounded in WHO ETAT/IMCI triage principles and clinical EWS literature.

---

## 🎯 End Result

This project provides a complete end-to-end Machine Learning pipeline and an interactive user interface for predicting patient deterioration.

**Interactive Streamlit Dashboard (`app.py`):**
- **User-Friendly Inputs:** Clinicians can input vitals (Temp, BP, HR, O2, etc.) via a web browser. (Supports both °C and °F).
- **Advanced Clinical Indices:** Automatically derives Shock Index, Pulse Pressure, and Mean Arterial Pressure (MAP).
- **Probability Breakdown:** Visual bar charts showing the exact confidence of the AI across 4 urgency levels (Low, Medium, High, Critical).
- **SHAP Explainable AI:** A dynamic breakdown of *why* the AI made its prediction, listing the top physiological factors driving the risk level.
- **Clinical Action Plans:** Maps the AI prediction to actionable clinical guidelines (e.g., Sepsis Six pathway, monitoring frequencies).

---

## 🚀 Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/Pranay623/Health-Risk-Triage-AI
cd health-risk-triage-ai
```

### 2. Install dependencies
Ensure you are using Python 3.9+ (Tested on 3.10 / 3.11).
```bash
pip install -r requirements.txt
```
*(Note: If you encounter `numpy` compatibility issues with pre-compiled libraries, the `requirements.txt` strictly enforces `numpy<2` to prevent crashes).*

### 3. Get the Data (PhysioNet 2019)
The raw PhysioNet 2019 data is not committed to this repo (40,336 PSV files, ~1GB). 

**How to download:**
1. Register at [physionet.org](https://physionet.org/content/challenge-2019/1.0.0/) and accept the data use agreement.
2. Download `training_setA.zip` and `training_setB.zip` using `wget` (with retry logic in case of network drops):
   ```bash
   wget -c --retry-connrefused --tries=5 https://physionet.org/files/challenge-2019/1.0.0/training/training_setA.zip
   wget -c --retry-connrefused --tries=5 https://physionet.org/files/challenge-2019/1.0.0/training/training_setB.zip
   ```
3. Extract both archives into `data/raw/physionet2019/`.

Expected structure after extraction:
```text
data/raw/physionet2019/
├── training_setA/
│   ├── p000001.psv
│   └── ...  (~20,000 files)
└── training_setB/
    ├── p100000.psv
    └── ...  (~20,000 files)
```

---

## 🧠 Running the Project

### Option A: Run the Interactive Web App (Frontend)
If the models are already trained (or downloaded), you can immediately spin up the interactive dashboard:
```bash
streamlit run app.py
```
This will open a local server at `http://localhost:8501` where you can input vitals and get real-time triage insights.

### Option B: Run the CLI Predictor
For terminal-based interactive testing:
```bash
python -m src.predict
```

### Option C: Run the Full ML Training Pipeline
To recreate the data processing, train the models, and generate SHAP explainability reports, run these self-contained stages in order:

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

## 📊 Major Specifications & Results

- **Model Performance:** XGBoost outperformed Random Forest, Decision Trees, MLPs, and Logistic Regression, achieving a **0.9423 F1-Macro** and **0.9983 AUROC** on the Phase 1 test slice.
- **Labeling Methodology:** A clinically grounded 4-level scale (Low, Medium, High, Critical) combining RCP NEWS2 2017 thresholds and Sepsis-3 ground truth overrides.
- **Explainability:** For lower urgencies, the model mirrors standard NEWS2 scoring. For 'Critical' cases, the model dynamically prioritizes age, ICU length of stay, and specific physiological interactions, proving it learns contextual deterioration rather than hardcoded cutoffs.

---

## ⚠️ Known Limitations

- **Row-level vs. Patient-level split:** Phase 1 uses row-level stratified splits. A rigorous patient-level train/test split is required before clinical validation (Planned for Phase 2).
- **Missing Data:** Temperature has ~65% missingness in the raw data; within-patient forward-fill + median fallback is used.
- **Consciousness Level:** AVPU/GCS scores are missing from PhysioNet 2019 but are critical for real-world NEWS2 scoring.
- **Model Bias:** Trained purely on ICU data; may not generalize perfectly to general ward or pre-hospital ambulatory triage without fine-tuning.

---

## 🔮 Phase 2 Roadmap

- [ ] Full 40,336-patient run and final benchmark with patient-level splits.
- [ ] Add symptom flags: chest pain, cough, breathlessness, AVPU/GCS.
- [ ] Syndrome-group classification (respiratory / circulatory / neurologic / infectious).
- [ ] Synthetic data augmentation for rare critical cases.
- [ ] LSTM/GRU time-series model (using ICULOS as temporal axis).

---

## 👤 Author

**Pranay** — B.Tech CSE (AI & ML), AKGEC Ghaziabad  
Built during summer research internship at DRDO DMSRDE, Kanpur