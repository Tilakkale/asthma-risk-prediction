from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "EducationLevel",
    "BMI",
    "Smoking",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "PollutionExposure",
    "PollenExposure",
    "DustExposure",
    "PetAllergy",
    "FamilyHistoryAsthma",
    "HistoryOfAllergies",
    "Eczema",
    "HayFever",
    "GastroesophagealReflux",
    "LungFunctionFEV1",
    "LungFunctionFVC",
    "Wheezing",
    "ShortnessOfBreath",
    "ChestTightness",
    "Coughing",
    "NighttimeSymptoms",
    "ExerciseInduced",
    "FEV1_FVC_ratio",
    "RespiratorySymptomScore",
    "AllergyExposureScore",
    "HighRiskHistory",
    "BMI_category",
    "PoorLifestyleRisk",
]


def compute_bmi_category(bmi: float) -> int:
    if bmi < 18.5:
        return 0
    if bmi < 25:
        return 1
    if bmi < 30:
        return 2
    return 3


def compute_symptom_score(df: pd.DataFrame) -> pd.Series:
    symptom_cols = [
        "Wheezing",
        "ShortnessOfBreath",
        "ChestTightness",
        "Coughing",
        "NighttimeSymptoms",
        "ExerciseInduced",
    ]
    return df[symptom_cols].sum(axis=1)


def compute_exposure_score(df: pd.DataFrame) -> pd.Series:
    exposure_cols = [
        "PollutionExposure",
        "PollenExposure",
        "DustExposure",
        "PetAllergy",
    ]
    return df[exposure_cols].sum(axis=1)


def compute_high_risk_history(df: pd.DataFrame) -> pd.Series:
    return (
        (df["FamilyHistoryAsthma"] == 1)
        | (df["HistoryOfAllergies"] == 1)
        | (df["Eczema"] == 1)
    ).astype(int)


def compute_poor_lifestyle_risk(df: pd.DataFrame) -> pd.Series:
    return (
        (df["PhysicalActivity"] < 3)
        | (df["DietQuality"] < 4)
        | (df["SleepQuality"] < 4)
    ).astype(int)


def standardize_gender(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Gender" not in df.columns:
        return df
    # Accept the numeric 0/1 values used in the dataset as well as common
    # string representations used by the Streamlit form.
    mapping = {"male": 1, "female": 0, "m": 1, "f": 0}
    text_values = df["Gender"].astype(str).str.strip().str.lower().map(mapping)
    numeric_values = pd.to_numeric(df["Gender"], errors="coerce")
    df["Gender"] = text_values.fillna(numeric_values).fillna(0).astype(int).clip(0, 1)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    target = df["Diagnosis"].copy() if "Diagnosis" in df.columns else None
    df = standardize_gender(df)

    required_columns = {
        "BMI",
        "LungFunctionFEV1",
        "LungFunctionFVC",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "Wheezing",
        "ShortnessOfBreath",
        "ChestTightness",
        "Coughing",
        "NighttimeSymptoms",
        "ExerciseInduced",
        "PollutionExposure",
        "PollenExposure",
        "DustExposure",
        "PetAllergy",
        "FamilyHistoryAsthma",
        "HistoryOfAllergies",
        "Eczema",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_columns)}")

    df["FEV1_FVC_ratio"] = df["LungFunctionFEV1"] / df["LungFunctionFVC"].replace(0, np.nan)
    df["FEV1_FVC_ratio"] = df["FEV1_FVC_ratio"].fillna(0)

    df["RespiratorySymptomScore"] = compute_symptom_score(df)
    df["AllergyExposureScore"] = compute_exposure_score(df)
    df["HighRiskHistory"] = compute_high_risk_history(df)
    df["BMI_category"] = df["BMI"].apply(compute_bmi_category)
    df["PoorLifestyleRisk"] = compute_poor_lifestyle_risk(df)

    df = df.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)
    if target is not None:
        df["Diagnosis"] = target.reset_index(drop=True)
    return df


def preprocess_data(csv_path: Path | str | None = None) -> pd.DataFrame:
    print("Loading Asthma Dataset...\n")

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "dataset" / "asthma.csv"

    df = pd.read_csv(csv_path)

    df = df.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")
    df = df.dropna()
    df = add_derived_features(df)

    print("\nPreprocessing Completed!\n")
    return df


def preprocess_input_data(raw_data: dict[str, object]) -> pd.DataFrame:
    df = pd.DataFrame([raw_data])
    df = df.copy()

    if "Gender" in df.columns:
        df = standardize_gender(df)

    df = add_derived_features(df)
    df = df.drop(columns=["PatientID", "DoctorInCharge"], errors="ignore")
    df = df.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0)
    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(df.head())
