"""
data_loading.py
================
Phase 1 data ingestion for the Health Risk Triage project.

Parses the raw PhysioNet 2019 Sepsis Challenge .psv files (one file per
patient, one row per ICU hour) into a single unified pandas DataFrame
containing ONLY the Phase 1 feature set:

    - Demographics : Age, Gender
    - Vitals       : HR, O2Sat, Temp, SBP, MAP, DBP, Resp
    - Time index   : ICULOS (hour count since ICU admission)
    - Label source : SepsisLabel (binary, used later by labeling.py)

The 26 lab-value columns (BUN, Lactate, WBC, etc.) are intentionally
dropped at load time. This is a deliberate Phase 1 scope decision, not
an oversight: the project targets vitals + demographics that are
"easily obtainable" in low-resource / first-contact settings, where
lab panels are typically unavailable. See README / project doc for the
full justification.

Each row in the output DataFrame represents one patient-hour, tagged
with a PatientID derived from the source filename (e.g. p001467.psv
-> "p001467"), since this identifier is not present in the PSV content
itself and would otherwise be lost.

Usage
-----
    from src.data_loading import load_physionet_phase1

    df = load_physionet_phase1("data/raw/physionet2019")
    df.to_parquet("data/processed/phase1/phase1_raw.parquet")

Or run directly:
    python src/data_loading.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------------------------------------------------
# Phase 1 column scope
# ----------------------------------------------------------------------
# Columns to keep from the raw 41-column PSV files. Order here is just
# for readability; actual parsing relies on the PSV header, not position,
# so this is robust even if PhysioNet ever changes column order.
PHASE1_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "Age",
    "Gender",
    "ICULOS",
    "SepsisLabel",
]

# Known full schema, kept here for reference / validation only.
FULL_PHYSIONET_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel",
]


def _find_psv_files(raw_dir: Path) -> list[Path]:
    """Recursively find all .psv files under raw_dir (training_setA + setB)."""
    files = sorted(raw_dir.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found under {raw_dir}. "
            "Check that download_physionet.py completed and files live "
            "under data/raw/physionet2019/training_setA (and _setB)."
        )
    return files


def _load_single_patient(filepath: Path) -> pd.DataFrame:
    """Load one patient's .psv file, keeping only Phase 1 columns."""
    df = pd.read_csv(filepath, sep="|")

    missing_cols = set(PHASE1_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{filepath.name} is missing expected columns: {missing_cols}")

    df = df[PHASE1_COLUMNS].copy()

    # PatientID from filename, e.g. "p001467.psv" -> "p001467"
    df.insert(0, "PatientID", filepath.stem)

    return df


def load_physionet_phase1(
    raw_dir: str | Path,
    limit: int | None = None,
    verbose_every: int = 5000,
) -> pd.DataFrame:
    """
    Load and concatenate all PhysioNet 2019 patient files into one
    Phase 1 DataFrame (one row per patient-hour).

    Parameters
    ----------
    raw_dir : str or Path
        Path to the folder containing training_setA/ and training_setB/
        (e.g. "data/raw/physionet2019").
    limit : int, optional
        If set, only load this many patient files. Useful for fast
        local testing before running on the full 40,336 patients.
    verbose_every : int
        Log progress every N files processed.

    Returns
    -------
    pd.DataFrame
        Unified dataframe with columns:
        PatientID, HR, O2Sat, Temp, SBP, MAP, DBP, Resp, Age, Gender,
        ICULOS, SepsisLabel
    """
    raw_dir = Path(raw_dir)
    files = _find_psv_files(raw_dir)

    if limit is not None:
        files = files[:limit]

    logger.info(f"Found {len(files)} .psv files. Beginning load...")

    frames = []
    failed = []

    for i, filepath in enumerate(files, start=1):
        try:
            frames.append(_load_single_patient(filepath))
        except Exception as e:
            failed.append((filepath.name, str(e)))

        if i % verbose_every == 0:
            logger.info(f"  processed {i}/{len(files)} files...")

    if failed:
        logger.warning(f"{len(failed)} files failed to load. First few: {failed[:5]}")

    if not frames:
        raise RuntimeError("No patient files loaded successfully — check failures above.")

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        f"Loaded {combined['PatientID'].nunique()} patients, "
        f"{len(combined):,} total patient-hour rows."
    )

    return combined


def summarize(df: pd.DataFrame) -> None:
    """Print a quick sanity-check summary of the loaded dataframe."""
    print("\n=== Phase 1 Dataset Summary ===")
    print(f"Rows (patient-hours): {len(df):,}")
    print(f"Unique patients:      {df['PatientID'].nunique():,}")
    print(f"Columns:              {list(df.columns)}")
    print(f"\nSepsisLabel positive rate: {df['SepsisLabel'].mean():.2%}")
    print(f"\nMissingness by column:")
    print((df.isna().mean() * 100).round(1).sort_values(ascending=False))
    print(f"\nAge range: {df['Age'].min():.1f} - {df['Age'].max():.1f}")
    print(f"Gender values: {sorted(df['Gender'].dropna().unique())}")


if __name__ == "__main__":
    # Quick local smoke test — load a small subset first, not all 40k,
    # so you can sanity check before committing to the full run.
    RAW_DIR = "data/raw/physionet2019"
    OUT_DIR = Path("data/processed/phase1")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_physionet_phase1(RAW_DIR, limit=None)  # set limit=500 for a fast test run
    summarize(df)

    out_path = OUT_DIR / "phase1_raw.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")