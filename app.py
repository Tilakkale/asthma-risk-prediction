import joblib
from pathlib import Path

import pandas as pd
import streamlit as st
import traceback

from preprocessing import preprocess_input_data

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_asthma_model.pkl"
AQI_PATH = PROJECT_ROOT / "dataset" / "aqi.csv"


def load_model() -> object:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_aqi_data() -> pd.DataFrame:
    df = pd.read_csv(AQI_PATH)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"])


def get_area_options(aqi_df: pd.DataFrame, state: str | None) -> list[str]:
    if state is None:
        return []
    areas = aqi_df[aqi_df["state"] == state]["area"].unique().tolist()
    return sorted(areas)


def get_aqi_summary(aqi_df: pd.DataFrame, state: str, area: str) -> pd.Series | None:
    subset = aqi_df[(aqi_df["state"] == state) & (aqi_df["area"] == area)]
    if subset.empty:
        return None
    latest = subset.sort_values("date", ascending=False).iloc[0]
    return latest


def risk_label(proba: float) -> str:
    if proba >= 0.7:
        return "High risk"
    if proba >= 0.4:
        return "Moderate risk"
    return "Low risk"


def build_patient_input() -> dict[str, object]:
    st.sidebar.header("Patient information")
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=35)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    ethnicity = st.sidebar.selectbox("Ethnicity", list(range(5)), index=0)
    education = st.sidebar.selectbox("Education level", list(range(5)), index=2)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
    smoking = st.sidebar.selectbox("Smoking history", [0, 1])
    physical_activity = st.sidebar.slider("Physical activity", min_value=0, max_value=10, value=5)
    diet_quality = st.sidebar.slider("Diet quality", min_value=0, max_value=10, value=6)
    sleep_quality = st.sidebar.slider("Sleep quality", min_value=0, max_value=10, value=6)
    pollution_exposure = st.sidebar.slider("Pollution exposure", min_value=0, max_value=10, value=5)
    pollen_exposure = st.sidebar.slider("Pollen exposure", min_value=0, max_value=10, value=4)
    dust_exposure = st.sidebar.slider("Dust exposure", min_value=0, max_value=10, value=3)
    pet_allergy = st.sidebar.selectbox("Pet allergy", [0, 1])
    family_history = st.sidebar.selectbox("Family history of asthma", [0, 1])
    history_of_allergies = st.sidebar.selectbox("History of allergies", [0, 1])
    eczema = st.sidebar.selectbox("Eczema", [0, 1])
    hayfever = st.sidebar.selectbox("Hay fever", [0, 1])
    gerd = st.sidebar.selectbox("Gastroesophageal reflux", [0, 1])
    fev1 = st.sidebar.number_input("Lung function FEV1", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    fvc = st.sidebar.number_input("Lung function FVC", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
    wheezing = st.sidebar.selectbox("Wheezing", [0, 1])
    breathlessness = st.sidebar.selectbox("Shortness of breath", [0, 1])
    chest_tightness = st.sidebar.selectbox("Chest tightness", [0, 1])
    coughing = st.sidebar.selectbox("Coughing", [0, 1])
    nighttime_symptoms = st.sidebar.selectbox("Nighttime symptoms", [0, 1])
    exercise_induced = st.sidebar.selectbox("Exercise-induced symptoms", [0, 1])

    return {
        "Age": age,
        "Gender": gender,
        "Ethnicity": ethnicity,
        "EducationLevel": education,
        "BMI": bmi,
        "Smoking": smoking,
        "PhysicalActivity": physical_activity,
        "DietQuality": diet_quality,
        "SleepQuality": sleep_quality,
        "PollutionExposure": pollution_exposure,
        "PollenExposure": pollen_exposure,
        "DustExposure": dust_exposure,
        "PetAllergy": pet_allergy,
        "FamilyHistoryAsthma": family_history,
        "HistoryOfAllergies": history_of_allergies,
        "Eczema": eczema,
        "HayFever": hayfever,
        "GastroesophagealReflux": gerd,
        "LungFunctionFEV1": fev1,
        "LungFunctionFVC": fvc,
        "Wheezing": wheezing,
        "ShortnessOfBreath": breathlessness,
        "ChestTightness": chest_tightness,
        "Coughing": coughing,
        "NighttimeSymptoms": nighttime_symptoms,
        "ExerciseInduced": exercise_induced,
    }


def main() -> None:
    st.set_page_config(page_title="Asthma Risk Assessment", layout="wide")
    st.title("Asthma Prediction and Environment Assessment")
    st.markdown(
        "Use the form to enter patient health data and view a professional asthma risk summary. "
        "The app also provides AQI environment risk information for a selected city."
    )

    model = load_model()
    aqi_df = load_aqi_data()

    with st.sidebar:
        st.header("Environment / AQI")
        state = st.selectbox("State", sorted(aqi_df["state"].unique().tolist()), index=0)
        areas = get_area_options(aqi_df, state)
        if areas:
            area = st.selectbox("Area", areas, index=0)
            aqi_summary = get_aqi_summary(aqi_df, state, area)
        else:
            area = None
            aqi_summary = None
            st.warning("No areas available for this state.")
        if aqi_summary is not None:
            st.metric("Latest AQI", int(aqi_summary["aqi_value"]))
            st.metric("Air quality", aqi_summary["air_quality_status"])
        else:
            st.warning("No AQI data available for this location.")

    patient_data = build_patient_input()

    st.subheader("Selected patient features")
    st.write(pd.DataFrame([patient_data]))

    if model is None:
        st.error(
            "Model not found. Run `python models/train_model.py` first to train and save the model."
        )
        return

    if st.button("Run assessment"):
        input_df = preprocess_input_data(patient_data)
        try:
            prediction = model.predict(input_df)[0]
            probability = None
            if hasattr(model, "predict_proba"):
                proba_values = model.predict_proba(input_df)
                if proba_values.shape[1] > 1:
                    probability = float(proba_values[0, 1])

            risk = risk_label(probability if probability is not None else float(prediction))

            cols = st.columns(3)
            cols[0].metric("Predicted asthma risk", "Positive" if prediction == 1 else "Negative")
            cols[1].metric("Risk level", risk)
            cols[2].metric("Positive probability", f"{probability * 100:.1f}%" if probability is not None else "N/A")

            st.markdown("### Derived health scores")
            st.write(
                input_df[
                    [
                        "FEV1_FVC_ratio",
                        "RespiratorySymptomScore",
                        "AllergyExposureScore",
                        "BMI_category",
                        "HighRiskHistory",
                        "PoorLifestyleRisk",
                    ]
                ]
            )

            if aqi_summary is not None:
                st.markdown("### Environment and AQI assessment")
                st.write(
                    {
                        "Location": f"{area}, {state}",
                        "AQI value": int(aqi_summary["aqi_value"]),
                        "Air quality status": aqi_summary["air_quality_status"],
                        "Prominent pollutant": aqi_summary["prominent_pollutants"],
                    }
                )
                if int(aqi_summary["aqi_value"]) >= 151:
                    st.warning("High air quality risk: use caution for asthma-sensitive patients.")
                elif int(aqi_summary["aqi_value"]) >= 101:
                    st.info("Moderate air quality risk: consider reducing outdoor exposure.")
                else:
                    st.success("Air quality is satisfactory for most patients.")
        except Exception as e:
            tb = traceback.format_exc()
            # Colorful animated error card using simple HTML + CSS and an animal emoji mascot
            error_html = """
<div style='border-radius:12px; padding:18px; background: linear-gradient(135deg,#ff8a00,#e52e71); color:white; box-shadow:0 8px 24px rgba(0,0,0,0.3)'>
  <div style='display:flex;align-items:center;gap:16px;'>
    <div style='font-size:48px; transform:translateY(0); animation:float 3s ease-in-out infinite;'>🐼</div>
    <div>
      <div style='font-weight:700; font-size:18px;'>Uh-oh — something went wrong</div>
      <div style='opacity:0.95; margin-top:6px;'>We couldn't complete the assessment due to an error. Details are below.</div>
      <div style='margin-top:8px; font-size:13px; opacity:0.95;'>Helpful steps: check model compatibility, retry, or contact the engineer.</div>
    </div>
  </div>
</div>
<style>
@keyframes float { 0% { transform: translateY(0);} 50% { transform: translateY(-8px);} 100% { transform: translateY(0);} }
.error-trace { background:#fff; color:#111; padding:10px 12px; border-radius:8px; margin-top:10px; overflow:auto; max-height:240px; }
</style>
"""

            st.markdown(error_html, unsafe_allow_html=True)
            with st.expander("Show technical details / traceback"):
                st.code(tb, language="text")
            # Suggest next actions in a colorful horizontal list
            st.markdown(
                """
<div style='display:flex;gap:10px;margin-top:10px;'>
  <div style='background:#0ea5e9;color:white;padding:8px 12px;border-radius:8px;font-weight:600;'>Retry assessment</div>
  <div style='background:#ef4444;color:white;padding:8px 12px;border-radius:8px;font-weight:600;'>Check model & features</div>
  <div style='background:#10b981;color:white;padding:8px 12px;border-radius:8px;font-weight:600;'>Run training</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("This app uses the trained asthma model and derived health features to produce a structured risk summary.")


if __name__ == "__main__":
    main()
