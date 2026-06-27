from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    # Robust mapping for various string dtypes and casings
    mapping = {"male": 1, "female": 0, "m": 1, "f": 0}
    # convert to string, normalize case and whitespace, then map
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map(mapping)
    # any unmapped -> treat as 0 (female) as a safe default, then cast to int
    df["Gender"] = df["Gender"].fillna(0).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = standardize_gender(df)

    df["FEV1_FVC_ratio"] = df["LungFunctionFEV1"] / df["LungFunctionFVC"].replace(0, np.nan)
    df["FEV1_FVC_ratio"] = df["FEV1_FVC_ratio"].fillna(0)

    df["RespiratorySymptomScore"] = compute_symptom_score(df)
    df["AllergyExposureScore"] = compute_exposure_score(df)
    df["HighRiskHistory"] = compute_high_risk_history(df)
    df["BMI_category"] = df["BMI"].apply(compute_bmi_category)
    df["PoorLifestyleRisk"] = compute_poor_lifestyle_risk(df)

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
    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(df.head())