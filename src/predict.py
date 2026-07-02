"""
predict.py
============
Interactive CLI to take user input for vitals and predict the patient's
urgency level and sepsis risk using the trained Phase 1 XGBoost model.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import yaml
import joblib
import pandas as pd
import numpy as np
import logging

# Suppress debug logs from features and labeling unless necessary
logging.getLogger("src.features").setLevel(logging.WARNING)
logging.getLogger("src.labeling").setLevel(logging.WARNING)

from src.labeling import assign_urgency_labels
from src.features import engineer_features
from src.preprocessing import CONTINUOUS_FEATURES

def main():
    print("="*60)
    print("   HEALTH RISK TRIAGE - INTERACTIVE PREDICTOR")
    print("="*60)
    print("Enter the patient's vitals. Press [Enter] to skip a vital if unknown.")
    print("Missing vitals will be imputed using population medians.")
    
    try:
        temp_in = input("\nTemperature (°C)       [e.g. 37.5] : ").strip()
        sbp_in  = input("Systolic BP (mmHg)     [e.g. 120]  : ").strip()
        dbp_in  = input("Diastolic BP (mmHg)    [e.g. 80]   : ").strip()
        hr_in   = input("Heart Rate (bpm)       [e.g. 85]   : ").strip()
        resp_in = input("Respiratory Rate       [e.g. 16]   : ").strip()
        o2_in   = input("SpO2 (%)               [e.g. 98]   : ").strip()
        age_in  = input("Age (years)            [e.g. 45]   : ").strip()
        
        # Convert to float or NaN
        temp = float(temp_in) if temp_in else np.nan
        sbp  = float(sbp_in)  if sbp_in  else np.nan
        dbp  = float(dbp_in)  if dbp_in  else np.nan
        hr   = float(hr_in)   if hr_in   else np.nan
        resp = float(resp_in) if resp_in else np.nan
        o2   = float(o2_in)   if o2_in   else np.nan
        age  = float(age_in)  if age_in  else 50.0  # Default age if completely unknown
        
        # 1. Create base DataFrame
        df_raw = pd.DataFrame([{
            "PatientID": "user_input",
            "HR": hr,
            "O2Sat": o2,
            "Temp": temp,
            "SBP": sbp,
            "MAP": np.nan,
            "DBP": dbp,
            "Resp": resp,
            "Age": age,
            "Gender": 0,       # default
            "ICULOS": 1,       # first hour
            "SepsisLabel": 0   # default (does not trigger override)
        }])
        
        # 2. Derive Labels (creates NEWS2Score)
        df_labeled = assign_urgency_labels(df_raw)
        
        # 3. Feature Engineering (creates ShockIndex, PulsePressure, AgeGroup, recovers MAP/DBP)
        df_features = engineer_features(df_labeled)
        
        # 4. Load Preprocessor
        prep_path = Path("data/processed/phase1/preprocessor.joblib")
        model_path = Path("models/xgboost_best.joblib")
        
        if not prep_path.exists() or not model_path.exists():
            print("\n[!] Error: Preprocessor or model not found. Run pipeline first.")
            sys.exit(1)
            
        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)
        
        scaler = preprocessor["scaler"]
        medians = preprocessor["medians"]
        
        # 5. Apply Imputation (Global medians)
        # We only apply medians to features that have missing values
        for col, val in medians.items():
            if col in df_features.columns and pd.isna(df_features[col].iloc[0]):
                df_features[col] = df_features[col].fillna(val)
                
        # 6. Select and order features for scaling
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
            
        feature_cols = config["features"]["model_features"]
        
        X = df_features[feature_cols].copy()
        
        # Scale continuous features
        scale_cols = [c for c in CONTINUOUS_FEATURES if c in feature_cols]
        X[scale_cols] = scaler.transform(X[scale_cols])
        
        # 7. Predict
        pred_class = model.predict(X)[0]
        pred_probs = model.predict_proba(X)[0]
        
        urgency_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
        urgency = urgency_map.get(pred_class, "Unknown")
        
        news2_score = df_labeled["NEWS2Score"].iloc[0]
        crit_prob = pred_probs[3] * 100
        
        # 8. Report Results
        print("\n" + "="*60)
        print("   PREDICTION RESULTS")
        print("="*60)
        print(f"  Predicted Urgency Level : {urgency.upper()}")
        print(f"  Calculated NEWS2 Score  : {news2_score} / 15")
        print(f"  Critical/Sepsis Risk    : {crit_prob:.1f}% probability of severe deterioration")
        
        print("\n=== CLINICAL INSIGHTS & ACTION PLAN ===")
        if urgency == "Critical" or crit_prob >= 50.0:
            print("[!] CRITICAL ALERT: Immediate stabilisation required.")
            print("    - Alert senior medical staff or Rapid Response Team.")
            print("    - Begin continuous physiological monitoring.")
            print("    - If sepsis suspected, start Sepsis Six pathway (IV fluids, O2, antibiotics).")
        elif urgency == "High" or (7 <= news2_score <= 8):
            print("[!] HIGH RISK: Emergency assessment required.")
            print("    - Urgent review by clinical team.")
            print("    - Increase monitoring frequency.")
        elif urgency == "Medium" or (5 <= news2_score <= 6):
            print("[-] MEDIUM RISK: Urgent review required.")
            print("    - Review by ward nurse/doctor.")
            print("    - Step up monitoring frequency (e.g., hourly).")
        else:
            print("[✓] LOW RISK: Routine monitoring.")
            print("    - Continue normal ward observations (e.g., 4-12 hourly).")
            print("    - Reassess if patient's condition changes.")
            
        print("\n=== SPECIFIC VITAL FLAGS ===")
        flags = []
        if pd.notna(sbp) and sbp <= 90:
            flags.append("Hypotension (Low SBP <= 90): High risk of shock. Consider fluid resuscitation.")
        if pd.notna(hr) and hr >= 111:
            flags.append("Tachycardia (HR >= 111): Sign of stress, infection, or dehydration.")
        if pd.notna(o2) and o2 <= 91:
            flags.append("Hypoxia (SpO2 <= 91%): Supplemental oxygen may be required.")
        if pd.notna(resp) and resp >= 25:
            flags.append("Tachypnea (Resp >= 25): Often the first sign of physiological deterioration.")
        if pd.notna(temp) and (temp >= 39.1 or temp <= 35.0):
            flags.append("Abnormal Temp: Possible systemic infection / sepsis.")
            
        if flags:
            for f in flags:
                print(f"  * {f}")
        else:
            print("  * No severe individual vital derangements detected.")
            
        print("="*60 + "\n")
        
    except ValueError:
        print("\n[!] Invalid input. Please enter numeric values only.")
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")

if __name__ == "__main__":
    main()
