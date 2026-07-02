import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path

# Suppress debug logs from backend modules
import logging
logging.getLogger("src.features").setLevel(logging.WARNING)
logging.getLogger("src.labeling").setLevel(logging.WARNING)

from src.labeling import assign_urgency_labels
from src.features import engineer_features
from src.preprocessing import CONTINUOUS_FEATURES

st.set_page_config(page_title="Health Risk Triage", page_icon="🏥", layout="centered")

st.title("🏥 Health Risk Triage Predictor")
st.markdown("Enter the patient's vitals below to predict the urgency level and sepsis risk. Missing values will be imputed using population medians.")

# Layout for inputs
st.subheader("Patient Vitals")
temp_unit = st.radio("Temperature Unit", ["°C", "°F"], horizontal=True)

col1, col2 = st.columns(2)

with col1:
    temp_placeholder = "e.g. 37.5" if temp_unit == "°C" else "e.g. 98.6"
    temp = st.number_input(f"Temperature ({temp_unit})", min_value=30.0, max_value=115.0, value=None, format="%.1f", placeholder=temp_placeholder)
    sbp = st.number_input("Systolic BP (mmHg)", min_value=30, max_value=250, value=None, placeholder="e.g. 120")
    hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=None, placeholder="e.g. 85")
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=None, placeholder="e.g. 45")

with col2:
    o2 = st.number_input("SpO2 (%)", min_value=50, max_value=100, value=None, placeholder="e.g. 98")
    dbp = st.number_input("Diastolic BP (mmHg)", min_value=20, max_value=150, value=None, placeholder="e.g. 80")
    resp = st.number_input("Respiratory Rate", min_value=5, max_value=60, value=None, placeholder="e.g. 16")

if st.button("Predict Urgency", type="primary", use_container_width=True):
    with st.spinner("Analyzing vitals..."):
        # Handle Nones and convert Fahrenheit to Celsius for the model
        temp_val = float(temp) if temp is not None else np.nan
        if temp_unit == "°F" and not np.isnan(temp_val):
            temp_val = (temp_val - 32) * 5.0 / 9.0
            
        sbp_val = float(sbp) if sbp is not None else np.nan
        dbp_val = float(dbp) if dbp is not None else np.nan
        hr_val = float(hr) if hr is not None else np.nan
        resp_val = float(resp) if resp is not None else np.nan
        o2_val = float(o2) if o2 is not None else np.nan
        age_val = float(age) if age is not None else 50.0

        # 1. Create base DataFrame
        df_raw = pd.DataFrame([{
            "PatientID": "user_input",
            "HR": hr_val,
            "O2Sat": o2_val,
            "Temp": temp_val,
            "SBP": sbp_val,
            "MAP": np.nan,
            "DBP": dbp_val,
            "Resp": resp_val,
            "Age": age_val,
            "Gender": 0,
            "ICULOS": 1,
            "SepsisLabel": 0
        }])

        # 2. Derive Labels (creates NEWS2Score)
        df_labeled = assign_urgency_labels(df_raw)
        
        # 3. Feature Engineering
        df_features = engineer_features(df_labeled)

        # 4. Load Preprocessor & Model
        prep_path = Path("data/processed/phase1/preprocessor.joblib")
        model_path = Path("models/xgboost_best.joblib")

        if not prep_path.exists() or not model_path.exists():
            st.error("Model or preprocessor not found. Please ensure the training pipeline was run.")
        else:
            preprocessor = joblib.load(prep_path)
            model = joblib.load(model_path)
            
            scaler = preprocessor["scaler"]
            medians = preprocessor["medians"]
            
            # 5. Apply Imputation
            for col, val in medians.items():
                if col in df_features.columns and pd.isna(df_features[col].iloc[0]):
                    df_features[col] = df_features[col].fillna(val)

            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            
            # 6. Prepare Final Feature Vector
            feature_cols = config["features"]["model_features"]
            X = df_features[feature_cols].copy()
            
            scale_cols = [c for c in CONTINUOUS_FEATURES if c in feature_cols]
            X[scale_cols] = scaler.transform(X[scale_cols])
            
            # 7. Predict
            pred_class = model.predict(X)[0]
            pred_probs = model.predict_proba(X)[0]
            
            urgency_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
            urgency = urgency_map.get(pred_class, "Unknown")
            news2_score = df_labeled["NEWS2Score"].iloc[0]
            crit_prob = pred_probs[3] * 100

            st.divider()
            st.subheader("Prediction Results")
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Urgency Level", urgency.upper())
            c2.metric("NEWS2 Score", f"{news2_score}/15")
            c3.metric("Critical/Sepsis Risk", f"{crit_prob:.1f}%")

            # Clinical Insights
            st.subheader("Clinical Insights & Action Plan")
            if urgency == "Critical" or crit_prob >= 50.0:
                st.error("**[!] CRITICAL ALERT: Immediate stabilisation required.**\n"
                         "- Alert senior medical staff or Rapid Response Team.\n"
                         "- Begin continuous physiological monitoring.\n"
                         "- If sepsis suspected, start Sepsis Six pathway (IV fluids, O2, antibiotics).")
            elif urgency == "High" or (7 <= news2_score <= 8):
                st.warning("**[!] HIGH RISK: Emergency assessment required.**\n"
                           "- Urgent review by clinical team.\n"
                           "- Increase monitoring frequency.")
            elif urgency == "Medium" or (5 <= news2_score <= 6):
                st.info("**[-] MEDIUM RISK: Urgent review required.**\n"
                        "- Review by ward nurse/doctor.\n"
                        "- Step up monitoring frequency (e.g., hourly).")
            else:
                st.success("**[✓] LOW RISK: Routine monitoring.**\n"
                           "- Continue normal ward observations (e.g., 4-12 hourly).\n"
                           "- Reassess if patient's condition changes.")
                           
            # Specific Flags
            st.subheader("Specific Vital Flags")
            flags = []
            if pd.notna(sbp_val) and sbp_val <= 90:
                flags.append("🚨 **Hypotension (Low SBP <= 90):** High risk of shock. Consider fluid resuscitation.")
            if pd.notna(hr_val) and hr_val >= 111:
                flags.append("🚨 **Tachycardia (HR >= 111):** Sign of stress, infection, or dehydration.")
            if pd.notna(o2_val) and o2_val <= 91:
                flags.append("🚨 **Hypoxia (SpO2 <= 91%):** Supplemental oxygen may be required.")
            if pd.notna(resp_val) and resp_val >= 25:
                flags.append("🚨 **Tachypnea (Resp >= 25):** Often the first sign of physiological deterioration.")
            if pd.notna(temp_val) and (temp_val >= 39.1 or temp_val <= 35.0):
                flags.append("🚨 **Abnormal Temp:** Possible systemic infection / sepsis.")
                
            if flags:
                for f in flags:
                    st.write(f)
            else:
                st.write("✅ No severe individual vital derangements detected.")
                
            st.divider()
            
            # --- ADVANCED INSIGHTS ---
            st.subheader("Advanced Clinical Indices")
            c4, c5, c6 = st.columns(3)
            
            shock_index = df_features["ShockIndex"].iloc[0]
            pulse_pressure = df_features["PulsePressure"].iloc[0]
            map_val = df_features["MAP"].iloc[0]
            
            si_status = "Normal" if shock_index < 0.7 else ("Caution" if shock_index <= 1.0 else "Unstable")
            c4.metric("Shock Index", f"{shock_index:.2f}" if pd.notna(shock_index) else "N/A", si_status, delta_color="off" if si_status=="Normal" else "inverse")
            
            pp_status = "Normal" if pd.notna(pulse_pressure) and 25 <= pulse_pressure <= 100 else "Abnormal"
            c5.metric("Pulse Pressure", f"{pulse_pressure:.1f} mmHg" if pd.notna(pulse_pressure) else "N/A", pp_status, delta_color="off" if pp_status=="Normal" else "inverse")
            
            map_status = "Normal" if pd.notna(map_val) and map_val >= 65 else "Low"
            c6.metric("MAP", f"{map_val:.1f} mmHg" if pd.notna(map_val) else "N/A", map_status, delta_color="off" if map_status=="Normal" else "inverse")
            
            # --- PROBABILITIES ---
            st.subheader("Model Probability Breakdown")
            prob_df = pd.DataFrame({
                "Urgency": ["Low", "Medium", "High", "Critical"],
                "Probability (%)": pred_probs * 100
            }).set_index("Urgency")
            st.bar_chart(prob_df)
            
            # --- SHAP EXPLAINABILITY ---
            import shap
            st.subheader(f"Why did the AI predict {urgency.upper()}?")
            with st.spinner("Calculating SHAP explanations..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                if isinstance(shap_values, list):
                    shap_values_3d = np.stack(shap_values, axis=2)
                else:
                    shap_values_3d = shap_values
                    
                # Extract SHAP values for the single patient and predicted class
                class_shap = shap_values_3d[0, :, pred_class]
                
                feature_impact = pd.DataFrame({
                    "Feature": X.columns,
                    "Impact": class_shap
                })
                
                # Sort by absolute impact
                feature_impact["Abs_Impact"] = feature_impact["Impact"].abs()
                feature_impact = feature_impact.sort_values(by="Abs_Impact", ascending=False).head(5)
                
                for _, row in feature_impact.iterrows():
                    if row["Abs_Impact"] > 0.05:  # Only show meaningful drivers
                        direction = "increased" if row["Impact"] > 0 else "decreased"
                        st.write(f"- **{row['Feature']}** {direction} the likelihood of this prediction.")
