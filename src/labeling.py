"""
labeling.py
============
Derives a 4-level urgency classification label for every patient-hour
row in the Phase 1 DataFrame produced by data_loading.py.

WHY THIS FILE EXISTS
---------------------
The PhysioNet 2019 dataset provides only a binary SepsisLabel (0/1).
That single binary column cannot serve as our target directly because:

  1. It covers only ONE clinical risk pathway (sepsis).
     Our project targets a broader urgency classification that includes
     all forms of physiological deterioration (cardiovascular, respiratory,
     shock, neurological), not just sepsis onset.

  2. The row-level positive rate is ~2-4%, making it extremely imbalanced
     for direct multi-class classification.

  3. A binary label gives us no gradient — we cannot distinguish a patient
     who is mildly unstable from one who is critically deteriorating.

SOLUTION: TWO-SIGNAL LABELING
-------------------------------
We derive urgency from TWO independent clinical signals, then merge them:

  Signal A — NEWS2-Style Vital Severity Score
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The National Early Warning Score 2 (NEWS2, Royal College of Physicians
  2017) is a validated aggregate scoring system used in UK NHS hospitals.
  It assigns 0-3 points per vital based on deviation from normal range,
  sums them into a total score, and maps the total to a risk band.

  We implement NEWS2 scoring for the 5 vitals available in Phase 1:
    - HR   (Heart Rate)
    - Resp (Respiratory Rate)
    - O2Sat (SpO2 — oxygen saturation)
    - SBP  (Systolic Blood Pressure)
    - Temp (Temperature)

  Note: NEWS2 also scores consciousness level (AVPU) and supplemental
  oxygen use, which are NOT available in Phase 1. We omit them and note
  this as a documented limitation. MAP/DBP are available but NEWS2 does
  not score them directly; they are kept as features for model training
  but not included in the urgency scoring rule. This is consistent with
  how NEWS2 is defined in the clinical literature.

  Threshold sources:
    Royal College of Physicians. National Early Warning Score (NEWS) 2.
    London: RCP, 2017. https://www.rcplondon.ac.uk/projects/outputs/
    national-early-warning-score-news-2

  Signal B — SepsisLabel Override
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  If a patient-hour has SepsisLabel = 1 (meaning the patient is within
  6 hours of clinical sepsis onset, per Sepsis-3 criteria), that hour
  is classified as CRITICAL regardless of the NEWS2 score.

  Clinical justification: Sepsis is a life-threatening organ dysfunction
  caused by a dysregulated host response to infection (Singer et al.,
  JAMA 2016). Its presence is itself a critical escalation trigger in
  WHO ETAT and all major triage frameworks, independent of whether vitals
  alone have deteriorated enough to score high.

URGENCY BANDS (final output)
------------------------------
  Label  | Integer | NEWS2 Score  | SepsisLabel
  -------|---------|--------------|------------
  Low    |    0    |    0 - 4     |      0
  Medium |    1    |    5 - 6     |      0
  High   |    2    |    7 - 8     |      0
  Critical|   3    |    >= 9  OR  |      1

  These bands mirror the NHS NEWS2 clinical response thresholds, adapted
  as a 4-level ordinal scale. The integer encoding (0-3) preserves
  ordinality for models that can exploit it, while the string labels
  are retained in a separate column for readability and reporting.

  Reference: NEWS2 clinical response thresholds (RCP 2017, Table 2):
    0-4  → Low    (routine monitoring)
    5-6  → Medium (urgent review)
    7+   → High   (emergency assessment)
    SepsisLabel=1 → Critical (immediate stabilisation)
    We split 7+ into High (7-8) and Critical (9+) to create a more
    granular 4-level scale aligned with the project's output spec.

HANDLING MISSING VITALS IN SCORING
-------------------------------------
NEWS2 scoring requires all 5 vitals. In our data:
  - Temp:  ~65% missing
  - O2Sat: ~13% missing
  - HR, Resp, SBP: 8-16% missing

When a vital is NaN for a given row, we score that parameter as 0
(normal) rather than treating the row as unscorable. This is a
conservative, deliberate choice:
  - It avoids discarding ~65% of rows due to Temp missingness alone.
  - It biases toward UNDER-estimating urgency (safer than over-alerting).
  - It will be partly corrected by forward-fill imputation in
    preprocessing.py before model training.
  - This limitation is documented here per TRIPOD Item 9 requirements.

Usage
-----
    from src.labeling import assign_urgency_labels

    df_labeled = assign_urgency_labels(df_phase1)

Or run directly after data_loading.py has saved its parquet:
    python src/labeling.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ======================================================================
# NEWS2 SCORING TABLES
# Source: Royal College of Physicians, NEWS2, 2017
# Each function takes a pandas Series of one vital and returns a
# Series of integer scores (0, 1, 2, or 3).
# NaN inputs return 0 (conservative — see module docstring).
# ======================================================================

def _score_hr(hr: pd.Series) -> pd.Series:
    """
    Heart Rate (beats per minute) — NEWS2 scoring.
    <=40        → 3
    41-50       → 1
    51-90       → 0  (normal)
    91-110      → 1
    111-130     → 2
    >=131       → 3
    """
    s = pd.Series(0, index=hr.index, dtype=int)
    s = s.where(hr.notna(), 0)             # NaN → 0
    s = s.where(~(hr <= 40), 3)
    s = s.where(~((hr >= 41) & (hr <= 50)), 1)
    s = s.where(~((hr >= 91) & (hr <= 110)), 1)
    s = s.where(~((hr >= 111) & (hr <= 130)), 2)
    s = s.where(~(hr >= 131), 3)
    return s


def _score_resp(resp: pd.Series) -> pd.Series:
    """
    Respiratory Rate (breaths per minute) — NEWS2 scoring.
    <=8         → 3
    9-11        → 1
    12-20       → 0  (normal)
    21-24       → 2
    >=25        → 3
    """
    s = pd.Series(0, index=resp.index, dtype=int)
    s = s.where(resp.notna(), 0)
    s = s.where(~(resp <= 8), 3)
    s = s.where(~((resp >= 9) & (resp <= 11)), 1)
    s = s.where(~((resp >= 21) & (resp <= 24)), 2)
    s = s.where(~(resp >= 25), 3)
    return s


def _score_o2sat(o2sat: pd.Series) -> pd.Series:
    """
    Oxygen Saturation / SpO2 (%) — NEWS2 Scale 1 scoring.
    (Scale 1 is used for patients NOT known to have hypercapnic
    respiratory failure. We use Scale 1 as default since we do not
    have that diagnosis flag in Phase 1.)
    <=91        → 3
    92-93       → 2
    94-95       → 1
    >=96        → 0  (normal)
    """
    s = pd.Series(0, index=o2sat.index, dtype=int)
    s = s.where(o2sat.notna(), 0)
    s = s.where(~(o2sat <= 91), 3)
    s = s.where(~((o2sat >= 92) & (o2sat <= 93)), 2)
    s = s.where(~((o2sat >= 94) & (o2sat <= 95)), 1)
    return s


def _score_sbp(sbp: pd.Series) -> pd.Series:
    """
    Systolic Blood Pressure (mmHg) — NEWS2 scoring.
    <=90        → 3
    91-100      → 2
    101-110     → 1
    111-219     → 0  (normal)
    >=220       → 3
    """
    s = pd.Series(0, index=sbp.index, dtype=int)
    s = s.where(sbp.notna(), 0)
    s = s.where(~(sbp <= 90), 3)
    s = s.where(~((sbp >= 91) & (sbp <= 100)), 2)
    s = s.where(~((sbp >= 101) & (sbp <= 110)), 1)
    s = s.where(~(sbp >= 220), 3)
    return s


def _score_temp(temp: pd.Series) -> pd.Series:
    """
    Temperature (°C) — NEWS2 scoring.
    <=35.0      → 3
    35.1-36.0   → 1
    36.1-38.0   → 0  (normal)
    38.1-39.0   → 1
    >=39.1      → 2
    """
    s = pd.Series(0, index=temp.index, dtype=int)
    s = s.where(temp.notna(), 0)
    s = s.where(~(temp <= 35.0), 3)
    s = s.where(~((temp > 35.0) & (temp <= 36.0)), 1)
    s = s.where(~((temp > 38.0) & (temp <= 39.0)), 1)
    s = s.where(~(temp > 39.0), 2)
    return s


# ======================================================================
# URGENCY BAND MAPPING
# ======================================================================

# Integer encoding preserves ordinality for ML models.
URGENCY_INT_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3,
}

URGENCY_LABEL_MAP = {v: k for k, v in URGENCY_INT_MAP.items()}


def _news2_score_to_band(score: pd.Series) -> pd.Series:
    """
    Map a NEWS2 total score (int) to urgency band string.
    0-4  → Low
    5-6  → Medium
    7-8  → High
    9+   → Critical
    """
    band = pd.Series("Low", index=score.index, dtype=object)
    band = band.where(~((score >= 5) & (score <= 6)), "Medium")
    band = band.where(~((score >= 7) & (score <= 8)), "High")
    band = band.where(~(score >= 9), "Critical")
    return band


# ======================================================================
# MAIN PUBLIC FUNCTION
# ======================================================================

def assign_urgency_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 4-level urgency labels for every patient-hour row.

    Adds three new columns to the input DataFrame:
      - NEWS2Score    : integer (0-15), sum of 5 vital sub-scores
      - UrgencyLabel  : string  ('Low', 'Medium', 'High', 'Critical')
      - UrgencyLevel  : integer (0, 1, 2, 3) — ordinal encoding of above

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_physionet_phase1(). Must contain columns:
        HR, Resp, O2Sat, SBP, Temp, SepsisLabel.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with three new columns appended.
        Input DataFrame is NOT modified in place.
    """
    df = df.copy()

    required = {"HR", "Resp", "O2Sat", "SBP", "Temp", "SepsisLabel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    # ---- Step 1: Compute per-vital NEWS2 sub-scores -------------------
    df["_score_HR"]   = _score_hr(df["HR"])
    df["_score_Resp"] = _score_resp(df["Resp"])
    df["_score_O2"]   = _score_o2sat(df["O2Sat"])
    df["_score_SBP"]  = _score_sbp(df["SBP"])
    df["_score_Temp"] = _score_temp(df["Temp"])

    # ---- Step 2: Total NEWS2 score (0-15) -----------------------------
    score_cols = ["_score_HR", "_score_Resp", "_score_O2", "_score_SBP", "_score_Temp"]
    df["NEWS2Score"] = df[score_cols].sum(axis=1).astype(int)

    # ---- Step 3: Map score to urgency band ----------------------------
    df["UrgencyLabel"] = _news2_score_to_band(df["NEWS2Score"])

    # ---- Step 4: SepsisLabel override → Critical ----------------------
    # If SepsisLabel = 1, force Critical regardless of NEWS2 score.
    # This is the two-signal merge described in the module docstring.
    sepsis_mask = df["SepsisLabel"] == 1
    df.loc[sepsis_mask, "UrgencyLabel"] = "Critical"

    # ---- Step 5: Integer encoding for ML models -----------------------
    df["UrgencyLevel"] = df["UrgencyLabel"].map(URGENCY_INT_MAP)

    # ---- Cleanup: drop internal scoring columns -----------------------
    df.drop(columns=score_cols, inplace=True)

    logger.info("Urgency labels assigned.")
    return df


# ======================================================================
# SUMMARY / VALIDATION
# ======================================================================

def summarize_labels(df: pd.DataFrame) -> None:
    """Print label distribution and validate no NaNs in label columns."""
    print("\n=== Urgency Label Distribution ===")

    counts = df["UrgencyLabel"].value_counts().reindex(
        ["Low", "Medium", "High", "Critical"], fill_value=0
    )
    total = len(df)
    for label, count in counts.items():
        bar = "#" * int(count / total * 50)
        print(f"  {label:<10} {count:>8,}  ({count/total*100:>5.1f}%)  {bar}")

    print(f"\n  Total rows : {total:,}")

    # How many of the Critical rows came from SepsisLabel vs NEWS2 alone?
    crit = df[df["UrgencyLabel"] == "Critical"]
    sepsis_driven   = (crit["SepsisLabel"] == 1).sum()
    news2_driven    = (crit["SepsisLabel"] == 0).sum()
    print(f"\n  Critical breakdown:")
    print(f"    SepsisLabel=1 override : {sepsis_driven:,}")
    print(f"    NEWS2 score >=9 only   : {news2_driven:,}")

    # Patient-level urgency peak (worst hour per patient)
    print("\n=== Patient-Level Peak Urgency ===")
    patient_peak = df.groupby("PatientID")["UrgencyLevel"].max()
    peak_counts = patient_peak.value_counts().sort_index()
    for lvl, cnt in peak_counts.items():
        label = URGENCY_LABEL_MAP[lvl]
        print(f"  {label:<10} (peak={lvl})  {cnt:>6,} patients  ({cnt/len(patient_peak)*100:.1f}%)")

    # Sanity: no NaN labels
    nan_labels = df["UrgencyLabel"].isna().sum()
    nan_levels = df["UrgencyLevel"].isna().sum()
    print(f"\n  NaN check — UrgencyLabel: {nan_labels}, UrgencyLevel: {nan_levels}")
    if nan_labels == 0 and nan_levels == 0:
        print("  [OK] No NaN labels — labeling complete.")
    else:
        print("  [WARN] WARNING: NaN labels found — investigate before proceeding.")


# ======================================================================
# DIRECT EXECUTION
# ======================================================================

if __name__ == "__main__":
    IN_PATH  = Path("data/processed/phase1/phase1_raw.parquet")
    OUT_PATH = Path("data/processed/phase1/phase1_labeled.parquet")

    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"{IN_PATH} not found. Run data_loading.py first."
        )

    logger.info(f"Loading {IN_PATH} ...")
    df = pd.read_parquet(IN_PATH)

    logger.info("Assigning urgency labels ...")
    df_labeled = assign_urgency_labels(df)

    summarize_labels(df_labeled)

    df_labeled.to_parquet(OUT_PATH, index=False)
    logger.info(f"Saved labeled dataset to {OUT_PATH}")