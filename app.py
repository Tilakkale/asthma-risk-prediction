"""
AsthmaAI — Single-Page Clinical Decision Support & Analytics
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocessing import MODEL_FEATURE_COLUMNS, preprocess_input_data

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_asthma_model.pkl"
PREP_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
AQI_PATH = PROJECT_ROOT / "dataset" / "aqi.csv"
REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_report.txt"
RESULTS_PATH = PROJECT_ROOT / "models" / "evaluation_results.json"

RISK_CONFIG = {
    "Low": dict(color="#34d399", bg="#ecfdf5", border="#86efac", threshold=0.35),
    "Moderate": dict(color="#f59e0b", bg="#fffbeb", border="#fcd34d", threshold=0.55),
    "High": dict(color="#ef4444", bg="#fef2f2", border="#fca5a5", threshold=0.75),
    "Critical": dict(color="#7c3aed", bg="#f5f3ff", border="#c4b5fd", threshold=1.01),
}

INPUT_DEFAULTS = {
    "age": 35, "bmi": 24.0, "gender": "Female", "ethnicity": 0,
    "fev1": 2.5, "fvc": 3.2, "education": 0, "smoking": "No",
    "activity": 5, "diet": 6, "sleep": 7, "pollution": 3, "pollen": 2,
    "dust": 3, "pet": "No", "family_history": "No", "allergies": "No",
    "eczema": "No", "hay_fever": "No", "reflux": "No", "wheezing": "No",
    "shortness": "No", "chest_tightness": "No", "coughing": "No",
    "nighttime": "No", "exercise_induced": "No",
}

def risk_tier(prob: float) -> str:
    for tier, cfg in RISK_CONFIG.items():
        if prob < cfg["threshold"]:
            return tier
    return "Critical"


st.set_page_config(page_title="Asthma Risk Analysis", page_icon=":material/monitor_heart:", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] { background: #f8fbff; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.1rem; max-width: 1460px; }
    #MainMenu, footer { visibility: hidden; }
    h1, h2, h3, h4 { color: #0f172a !important; }
    p, label, span, div { color: #334155; }

    .hero {
        background: linear-gradient(120deg, #e0f2fe 0%, #f8fafc 55%, #dbeafe 100%);
        border: 1px solid #bfdbfe; border-radius: 18px; padding: 24px 26px; margin-bottom: 20px;
    }
    .hero .h-title { font-size: 28px; font-weight: 800; color: #0f172a; }
    .hero .h-sub { font-size: 14px; color: #475569; margin-top: 6px; max-width: 880px; line-height: 1.6; }
    .hero .h-chip { display: inline-block; margin-top: 12px; margin-right: 8px; padding: 5px 12px; background: #dbeafe; border: 1px solid #93c5fd; border-radius: 999px; color: #1d4ed8; font-size: 12px; font-weight: 700; }

    .kpi-card {
        background: #ffffff; border: 1px solid #dbe7f3; border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }
    .kpi-card .k-label { font-size: 11px; color: #64748b; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-card .k-value { font-size: 28px; font-weight: 800; color: #0f172a; margin-top: 4px; }
    .kpi-card .k-sub { font-size: 12px; color: #64748b; margin-top: 2px; }

    .section {
        display: flex; align-items: center; gap: 10px; padding: 12px 16px; margin: 28px 0 14px;
        background: #ffffff; border: 1px solid #dbe7f3; border-radius: 10px; box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }
    .section .s-num {
        width: 26px; height: 26px; border-radius: 8px; background: linear-gradient(135deg,#2563eb,#0ea5e9);
        color: #fff; font-size: 13px; font-weight: 800; display: flex; align-items: center; justify-content: center;
    }
    .section .s-title { font-size: 15px; font-weight: 800; color: #0f172a; }
    .section .s-caption { font-size: 12px; color: #64748b; margin-left: auto; }

    .risk-banner { border-radius: 14px; padding: 18px 22px; border: 1px solid; margin-bottom: 18px; }
    .risk-banner .rb-title { font-size: 24px; font-weight: 800; }
    .risk-banner .rb-sub { font-size: 13px; margin-top: 3px; line-height: 1.6; }

    .rec-box {
        background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #0ea5e9; border-radius: 12px;
        padding: 16px 18px; font-size: 13px; color: #334155; line-height: 1.7; margin-top: 10px;
    }

    .assistant-card { background: #f8fbff; border: 1px solid #bfdbfe; border-radius: 16px; padding: 22px 26px; }
    .assistant-card .as-head { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
    .assistant-card .as-dot { width: 10px; height: 10px; border-radius: 50%; background: #16a34a; box-shadow: 0 0 10px #16a34a; }
    .assistant-card .as-title { font-size: 15px; font-weight: 800; color: #0f172a; }
    .assistant-card .as-chip { margin-left: auto; font-size: 11px; font-weight: 800; padding: 4px 12px; border-radius: 999px; }
    .as-row { padding: 10px 0; border-bottom: 1px solid #e2e8f0; font-size: 13px; line-height: 1.7; color: #334155; }
    .as-row:last-child { border-bottom: none; }
    .as-row .as-lbl { display: block; margin-bottom: 4px; font-size: 11px; font-weight: 800; color: #2563eb; text-transform: uppercase; letter-spacing: 0.06em; }
    .as-row .good { color: #059669; font-weight: 700; }
    .as-row .bad { color: #dc2626; font-weight: 700; }
    .as-row .warn { color: #d97706; font-weight: 700; }

    table.assess-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .assess-table th { text-align: left; padding: 10px 12px; color: #64748b; font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase; border-bottom: 1px solid #e2e8f0; background: #f8fafc; }
    .assess-table td { padding: 9px 12px; border-bottom: 1px solid #e2e8f0; color: #334155; vertical-align: middle; }
    .assess-table tr.cat-row td { background: #eff6ff; color: #1d4ed8; font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .status-pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 11px; font-weight: 800; }
    .pill-normal { background: #dcfce7; color: #166534; }
    .pill-atrisk { background: #fef3c7; color: #92400e; }
    .pill-abnormal { background: #fee2e2; color: #991b1b; }
    .impact-wrap { display: flex; align-items: center; gap: 10px; }
    .impact-bar { background: #e2e8f0; border-radius: 5px; height: 8px; width: 95px; overflow: hidden; }
    .impact-fill { height: 100%; border-radius: 5px; }
    .impact-txt { font-size: 11px; font-weight: 700; }

    [data-testid="stNumberInput"] input, [data-testid="stSlider"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background: #ffffff !important; color: #0f172a !important; border-color: #cbd5e1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_resource
def load_preprocessor():
    if PREP_PATH.exists():
        return joblib.load(PREP_PATH)
    return None

@st.cache_data
def load_aqi() -> pd.DataFrame:
    if not AQI_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(AQI_PATH)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)

@st.cache_data
def load_report() -> dict:
    metrics = {}
    if REPORT_PATH.exists():
        try:
            for line in REPORT_PATH.read_text(encoding="utf-8").splitlines():
                for key in ("Accuracy", "Positive precision", "Positive recall", "Positive F1", "Average precision", "ROC-AUC"):
                    if line.startswith(f"{key}:"):
                        metrics[key] = line.split(":", 1)[1].strip()
                # Parse confusion matrix from evaluate_model.py format
                if line.startswith("TN="):
                    parts = line.split()
                    for p in parts:
                        if p.startswith("TN="):
                            metrics["TN"] = p.split("=")[1]
                        elif p.startswith("FP="):
                            metrics["FP"] = p.split("=")[1]
                elif line.startswith("FN="):
                    parts = line.split()
                    for p in parts:
                        if p.startswith("FN="):
                            metrics["FN"] = p.split("=")[1]
                        elif p.startswith("TP="):
                            metrics["TP"] = p.split("=")[1]
                # Parse confusion matrix from train_model.py format
                if line.startswith("Confusion matrix:"):
                    cm_part = line.split(":", 1)[1].strip()
                    for item in cm_part.split():
                        if "=" in item:
                            k, v = item.split("=", 1)
                            metrics[k] = v
        except Exception:
            pass
    return metrics


@st.cache_data
def load_evaluation_results() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_research_results(results: dict) -> None:
    with st.expander("Research evaluation", expanded=False):
        if not results:
            st.info("Run python models/comprehensive_evaluation.py to generate research results.")
            return

        status = results.get("validation_status", {})
        model_metrics = results.get("model", {})
        ranking = results.get("ranking_metrics", {})
        baseline = results.get("majority_baseline", {})
        calibration = results.get("calibration", {})
        st.markdown(f"**Validation status:** {status.get('status', 'Not available')}")
        metric_columns = st.columns(6)
        values = [
            ("Model accuracy", model_metrics.get("accuracy"), "%"),
            ("Baseline accuracy", baseline.get("accuracy"), "%"),
            ("ROC-AUC", ranking.get("roc_auc"), "%"),
            ("PR-AUC", ranking.get("pr_auc"), "%"),
            ("F1-score", model_metrics.get("f1"), ""),
            ("Brier score", calibration.get("brier_score"), ""),
        ]
        for column, (label, value, suffix) in zip(metric_columns, values):
            if value is None:
                display_value = "-"
            elif suffix == "%":
                display_value = f"{float(value) * 100:.2f}%"
            else:
                display_value = f"{float(value):.3f}"
            column.metric(label, display_value)

        st.caption("The saved evaluation uses a fixed stratified hold-out test set. Metrics are research results, not clinical validation.")
        plot_columns = st.columns(3)
        plot_files = [
            ("ROC curve", "evaluation_roc_curve.png"),
            ("Precision-recall curve", "evaluation_pr_curve.png"),
            ("Calibration curve", "evaluation_calibration_curve.png"),
        ]
        for column, (caption, filename) in zip(plot_columns, plot_files):
            path = PROJECT_ROOT / "models" / filename
            if path.exists():
                column.image(str(path), caption=caption, use_container_width=True)

        results_json = json.dumps(results, indent=2, default=str).encode("utf-8")
        st.download_button(
            "Download evaluation results (JSON)",
            data=results_json,
            file_name="asthma_evaluation_results.json",
            mime="application/json",
        )


def compute_derived(p: dict) -> dict:
    fev1_fvc = round(p["LungFunctionFEV1"] / p["LungFunctionFVC"], 3) if p["LungFunctionFVC"] > 0 else 0.0
    resp = sum([
        p["Wheezing"], p["ShortnessOfBreath"], p["ChestTightness"],
        p["Coughing"], p["NighttimeSymptoms"], p["ExerciseInduced"]
    ])
    allergy = sum([
        p.get("PollutionExposure", 0),
        p.get("PollenExposure", 0),
        p.get("DustExposure", 0),
        p.get("PetAllergy", 0),
    ])
    bmi_val = p["BMI"]
    bmi_cat = "Underweight" if bmi_val < 18.5 else "Normal" if bmi_val < 25 else "Overweight" if bmi_val < 30 else "Obese"
    high_risk = int(p["FamilyHistoryAsthma"] or p["HistoryOfAllergies"] or p["Eczema"])
    poor_life = int(p["PhysicalActivity"] < 3 or p["DietQuality"] < 4 or p["SleepQuality"] < 4)
    return dict(fev1_fvc=fev1_fvc, resp=resp, allergy=allergy, bmi_cat=bmi_cat, high_risk=high_risk, poor_life=poor_life)

AQI_LEVELS = [
    (50, "Good", "#34d399"),
    (100, "Satisfactory", "#a3e635"),
    (200, "Moderate", "#fbbf24"),
    (300, "Poor", "#fb923c"),
    (400, "Very Poor", "#f87171"),
    (500, "Severe / Hazardous", "#c084fc"),
]


def aqi_label(val: float) -> tuple[str, str]:
    for limit, label, color in AQI_LEVELS:
        if val <= limit:
            return label, color
    return "Severe / Hazardous", "#c084fc"


def risk_gauge(prob: float) -> go.Figure:
    pct = round(prob * 100, 1)
    tier = risk_tier(prob)
    cfg = RISK_CONFIG[tier]
    fig = go.Figure(go.Indicator(mode="gauge+number", value=pct, number=dict(suffix="%", font=dict(size=40, color=cfg["color"])), gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor="#64748b", tickvals=[0, 35, 55, 75, 100], ticktext=["0%", "35%", "55%", "75%", "100%"]), bar=dict(color=cfg["color"], thickness=0.32), bgcolor="#ffffff", borderwidth=0, steps=[dict(range=[0, 35], color="rgba(52,211,153,0.16)"), dict(range=[35, 55], color="rgba(251,191,36,0.16)"), dict(range=[55, 75], color="rgba(248,113,113,0.16)"), dict(range=[75, 100], color="rgba(192,132,252,0.16)")], threshold=dict(line=dict(color=cfg["color"], width=3), thickness=0.85, value=pct))))
    fig.update_layout(height=270, margin=dict(t=20, b=0, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a", size=12))
    return fig


def radar_chart(patient: dict, derived: dict) -> go.Figure:
    ratio = derived["fev1_fvc"]
    resp = derived["resp"]
    act, diet, sleep = patient["PhysicalActivity"], patient["DietQuality"], patient["SleepQuality"]
    poll, pollen, dust, pet = patient["PollutionExposure"], patient["PollenExposure"], patient["DustExposure"], patient["PetAllergy"]
    bmi = patient["BMI"]
    pulmonary = float(np.clip((ratio - 0.40) / (0.85 - 0.40) * 100, 0, 100))
    symptoms = float(np.clip(100 - resp * 16.67, 0, 100))
    lifestyle = float((act / 10 * 100 + diet / 10 * 100 + sleep / 10 * 100) / 3)
    exposure = float(np.clip(100 - (poll + pollen + dust + pet * 10) / 40 * 100, 0, 100))
    history = 100.0 if derived["high_risk"] == 0 else 25.0
    if 18.5 <= bmi <= 25:
        body = 100.0
    elif bmi < 18.5:
        body = float(np.clip(100 - (18.5 - bmi) * 20, 30, 100))
    else:
        body = float(np.clip(100 - (bmi - 25) * 6, 20, 100))
    cats = ["Pulmonary", "Symptoms", "Lifestyle", "Exposure", "History", "Body composition"]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[pulmonary, symptoms, lifestyle, exposure, history, body], theta=cats, fill="toself", name="Patient", line=dict(color="#2563eb", width=2.5), fillcolor="rgba(37,99,235,0.18)"))
    fig.add_trace(go.Scatterpolar(r=[100, 100, 100, 100, 100, 100], theta=cats, fill="toself", name="Healthy baseline", line=dict(color="#059669", width=2, dash="dot"), fillcolor="rgba(5,150,105,0.10)"))
    fig.add_trace(go.Scatterpolar(r=[40, 40, 40, 40, 40, 40], theta=cats, fill="toself", name="Risk threshold", line=dict(color="#ef4444", width=2, dash="dash"), fillcolor="rgba(239,68,68,0.06)"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#dbe7f3", tickcolor="#64748b"), bgcolor="rgba(0,0,0,0)", angularaxis=dict(gridcolor="#dbe7f3", tickfont=dict(color="#334155", size=11))), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(size=11)), title=dict(text="Patient profile vs healthy baseline", font=dict(size=13, color="#0f172a")), height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#0f172a", size=12), margin=dict(t=40, b=40, l=40, r=40))
    return fig


def _st(value, normal, atrisk):
    if isinstance(normal, tuple):
        lo, hi = normal
        if isinstance(atrisk, tuple):
            alo, ahi = atrisk
            if lo <= value <= hi:
                return "Normal"
            if alo <= value <= ahi:
                return "At-risk"
            return "Abnormal"
    else:
        if value == normal:
            return "Normal"
        if callable(atrisk) and atrisk(value):
            return "At-risk"
        return "Abnormal"
    return "Abnormal"


def build_assessment(patient: dict, derived: dict, shap_map: dict) -> list[dict]:
    rows = []
    def add(label, category, healthy, value, status, shap):
        rows.append(dict(label=label, category=category, healthy=healthy, value=value, status=status, shap=shap))
    add("Age", "Body & Demographics", "12 - 65 years", f"{patient['Age']} yr", _st(patient["Age"], (12, 65), (65, 80)), shap_map.get("Age", 0.0))
    add("BMI", "Body & Demographics", "18.5 - 24.9", f"{patient['BMI']:.1f}", _st(patient["BMI"], (18.5, 24.9), (25.0, 29.9)), shap_map.get("BMI", 0.0))
    fev1 = patient["LungFunctionFEV1"]
    fvc = patient["LungFunctionFVC"]
    add("FEV1", "Lung Function", "≥ 2.5 L", f"{fev1:.2f} L", _st(fev1, (2.5, 8.0), (1.5, 2.5)), shap_map.get("LungFunctionFEV1", 0.0))
    add("FVC", "Lung Function", "≥ 3.0 L", f"{fvc:.2f} L", _st(fvc, (3.0, 8.0), (2.0, 3.0)), shap_map.get("LungFunctionFVC", 0.0))
    add("FEV1 / FVC ratio", "Lung Function", "≥ 0.70", f"{derived['fev1_fvc']:.2f}", _st(derived["fev1_fvc"], (0.70, 1.0), (0.60, 0.70)), shap_map.get("FEV1_FVC_ratio", 0.0))
    add("Physical activity", "Lifestyle", "≥ 4 days/week", f"{patient['PhysicalActivity']} days", _st(patient["PhysicalActivity"], (4, 10), (2, 3)), shap_map.get("PhysicalActivity", 0.0))
    add("Diet quality", "Lifestyle", "≥ 7 / 10", f"{patient['DietQuality']} / 10", _st(patient["DietQuality"], (7, 10), (4, 6)), shap_map.get("DietQuality", 0.0))
    add("Sleep quality", "Lifestyle", "≥ 7 / 10", f"{patient['SleepQuality']} / 10", _st(patient["SleepQuality"], (7, 10), (4, 6)), shap_map.get("SleepQuality", 0.0))
    add("Smoking", "Lifestyle", "No", "Yes" if patient["Smoking"] else "No", "Normal" if not patient["Smoking"] else "Abnormal", shap_map.get("Smoking", 0.0))
    sym_map = [("Wheezing", "Wheezing"), ("Shortness of breath", "ShortnessOfBreath"), ("Chest tightness", "ChestTightness"), ("Coughing", "Coughing"), ("Nighttime symptoms", "NighttimeSymptoms"), ("Exercise-induced", "ExerciseInduced")]
    for label, key in sym_map:
        add(label, "Symptoms", "No", "Yes" if patient[key] else "No", "Normal" if not patient[key] else "Abnormal", shap_map.get(key, 0.0))
    add("Symptom score", "Symptoms", "0 active", f"{derived['resp']} / 6", _st(derived["resp"], (0, 0), (1, 2)), shap_map.get("RespiratorySymptomScore", 0.0))
    add("Pollution exposure", "Exposure", "0 - 3 / 10", f"{patient['PollutionExposure']} / 10", _st(patient["PollutionExposure"], (0, 3), (4, 6)), shap_map.get("PollutionExposure", 0.0))
    add("Pollen exposure", "Exposure", "0 - 3 / 10", f"{patient['PollenExposure']} / 10", _st(patient["PollenExposure"], (0, 3), (4, 6)), shap_map.get("PollenExposure", 0.0))
    add("Dust exposure", "Exposure", "0 - 3 / 10", f"{patient['DustExposure']} / 10", _st(patient["DustExposure"], (0, 3), (4, 6)), shap_map.get("DustExposure", 0.0))
    add("Pet allergy", "Exposure", "No", "Yes" if patient["PetAllergy"] else "No", "Normal" if not patient["PetAllergy"] else "Abnormal", shap_map.get("PetAllergy", 0.0))
    hist_map = [("Family history of asthma", "FamilyHistoryAsthma"), ("History of allergies", "HistoryOfAllergies"), ("Eczema", "Eczema"), ("Hay fever", "HayFever"), ("Gastro-oesophageal reflux", "GastroesophagealReflux")]
    for label, key in hist_map:
        add(label, "Medical History", "No", "Yes" if patient[key] else "No", "Normal" if not patient[key] else "Abnormal", shap_map.get(key, 0.0))
    return rows


def assessment_table_html(rows: list[dict]) -> str:
    max_abs = max((abs(r["shap"]) for r in rows), default=1.0) or 1.0
    categories = []
    for r in rows:
        if r["category"] not in categories:
            categories.append(r["category"])
    pill = {"Normal": "pill-normal", "At-risk": "pill-atrisk", "Abnormal": "pill-abnormal"}
    html = ['<div style="overflow-x:auto"><table class="assess-table">']
    html.append("<thead><tr><th>Parameter</th><th>Value</th><th>Healthy range</th><th>Status</th><th>Impact on risk</th></tr></thead><tbody>")
    for cat in categories:
        html.append(f'<tr class="cat-row"><td colspan="5">{cat}</td></tr>')
        for r in rows:
            if r["category"] != cat:
                continue
            width = min(abs(r["shap"]) / max_abs * 100, 100)
            direction = "Increases" if r["shap"] > 0 else "Decreases" if r["shap"] < 0 else "Neutral"
            color = "#ef4444" if r["shap"] > 0 else "#059669" if r["shap"] < 0 else "#64748b"
            html.append(f'<tr><td>{r["label"]}</td><td style="font-weight:700;color:#0f172a">{r["value"]}</td><td style="color:#64748b">{r["healthy"]}</td><td><span class="status-pill {pill[r["status"]]}">{r["status"]}</span></td><td><div class="impact-wrap"><div class="impact-bar"><div class="impact-fill" style="width:{width:.0f}%;background:{color}"></div></div><span class="impact-txt" style="color:{color}">{direction}</span></div></td></tr>')
    html.append("</tbody></table></div>")
    return "".join(html)


def generate_assistant(tier: str, prob: float, rows: list[dict], shap_map: dict) -> str:
    abnormal = [r for r in rows if r["status"] == "Abnormal"]
    atrisk = [r for r in rows if r["status"] == "At-risk"]
    normal = [r for r in rows if r["status"] == "Normal"]
    sorted_shap = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    drivers = [k.replace("_", " ").title() for k, v in sorted_shap if v > 0][:4]
    protective = [k.replace("_", " ").title() for k, v in sorted_shap if v < 0][:3]
    cfg = RISK_CONFIG[tier]
    chip_style = f"background:{cfg['bg']};color:{cfg['color']};border:1px solid {cfg['border']}"
    parts = []
    parts.append(f'<div class="as-head"><div class="as-dot"></div><div class="as-title">AI Assistant — interpretation summary</div><span class="as-chip" style="{chip_style}">{tier} risk · {prob * 100:.1f}%</span></div>')
    parts.append(f'<div class="as-row"><span class="as-lbl">Profile summary</span>Of {len(rows)} assessed parameters, <span class="bad">{len(abnormal)} are abnormal</span>, <span class="warn">{len(atrisk)} are borderline</span> and <span class="good">{len(normal)} are within healthy range</span>.</div>')
    if drivers:
        parts.append(f'<div class="as-row"><span class="as-lbl">Primary risk drivers</span><span class="bad">{", ".join(drivers)}</span> — these features contribute most to the elevated risk probability and should be the focus of intervention.</div>')
    if protective:
        parts.append(f'<div class="as-row"><span class="as-lbl">Protective findings</span><span class="good">{", ".join(protective)}</span> — normal values in these areas help lower overall risk.</div>')
    if abnormal:
        names = ", ".join(r["label"] for r in abnormal[:4])
        parts.append(f'<div class="as-row"><span class="as-lbl">Needs attention</span>Abnormal parameters: <span class="bad">{names}</span>. Prioritise these in the management plan.</div>')
    tier_recs = {
        "Low": "Maintain current lifestyle. Schedule an annual review of lung function and keep monitoring the factors listed above.",
        "Moderate": "Schedule a pulmonary function follow-up. Review inhaler technique, allergen avoidance, and implement an asthma action plan.",
        "High": "Refer to a pulmonologist promptly. Ensure a rescue inhaler is prescribed, an asthma action plan is in place, and trigger avoidance is enforced.",
        "Critical": "Urgent specialist referral required. Initiate immediate respiratory assessment and do not delay intervention.",
    }
    parts.append(f'<div class="as-row"><span class="as-lbl">Recommended actions</span>{tier_recs[tier]}</div>')
    return f'<div class="assistant-card">{"".join(parts)}</div>'


def recommendation(tier: str, derived: dict) -> str:
    drivers = []
    if derived["fev1_fvc"] < 0.70:
        drivers.append("low FEV1/FVC ratio (airflow obstruction)")
    if derived["resp"] >= 4:
        drivers.append("multiple respiratory symptoms")
    if derived["high_risk"] == 1:
        drivers.append("positive family history or allergy background")
    if derived["poor_life"] == 1:
        drivers.append("lifestyle risk factors (low activity, poor diet/sleep)")
    driver_str = "; ".join(drivers) if drivers else "clinical and environmental factors"
    recs = {
        "Low": f"Low risk detected. Maintain current lifestyle. Monitor {driver_str} at annual review.",
        "Moderate": f"Moderate risk. Schedule a pulmonary function follow-up. Key drivers: {driver_str}.",
        "High": f"High risk. Refer to a pulmonologist promptly. Primary drivers: {driver_str}.",
        "Critical": f"Critical risk. Urgent specialist referral required. Primary drivers: {driver_str}.",
    }
    return recs[tier]


def apply_profile(profile: dict) -> None:
    """Set form state before widgets are created, then refresh the page."""
    for key, value in profile.items():
        st.session_state[key] = value
    st.session_state.pop("last_result", None)


def input_toolbar() -> None:
    """Provide a way to reset the assessment form."""
    if st.button("Reset form", use_container_width=False):
        apply_profile(INPUT_DEFAULTS)
        st.rerun()


def yn(label: str, key: str, help: str = "") -> int:
    result = st.radio(label, ["No", "Yes"], key=key, horizontal=True, help=help)
    return 1 if result == "Yes" else 0


def drivers_chart(shap_map: dict, top_n: int = 8):
    items = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    items = [it for it in items if abs(it[1]) > 1e-6]
    if not items:
        return None
    names = [i[0].replace("_", " ") for i in items]
    vals = [i[1] for i in items]
    colors = ["#ef4444" if v > 0 else "#059669" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in vals], textposition="outside"))
    fig.update_layout(title=dict(text="Top risk drivers in this prediction", font=dict(size=13, color="#0f172a")), xaxis=dict(title="SHAP value (positive = raises risk)", gridcolor="#dbe7f3"), yaxis=dict(title=""), height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155", size=12), margin=dict(t=40, b=20, l=10, r=70))
    return fig


def shap_full_chart(model, X_df: pd.DataFrame):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_df)
    except Exception:
        try:
            import shap
            explainer = shap.LinearExplainer(model, X_df)
            sv = explainer.shap_values(X_df)
        except Exception:
            return {}, None
    if isinstance(sv, list):
        sv = sv[1]
    values = np.asarray(sv)
    if values.ndim == 3:
        # Newer SHAP versions return (rows, features, classes).
        values = values[:, :, 1]
    if values.ndim == 2:
        values = values[0]
    vals = values.reshape(-1)
    if len(vals) != len(X_df.columns):
        return {}, None
    shap_map = {col: float(v) for col, v in zip(X_df.columns, vals)}
    df = (pd.DataFrame({"Feature": X_df.columns.tolist(), "SHAP": vals}).assign(abs_shap=lambda d: d["SHAP"].abs()).sort_values("abs_shap").tail(14))
    df["color"] = df["SHAP"].apply(lambda v: "#ef4444" if v > 0 else "#059669")
    fig = go.Figure(go.Bar(x=df["SHAP"], y=df["Feature"], orientation="h", marker_color=df["color"].tolist(), text=df["SHAP"].apply(lambda v: f"{v:+.3f}"), textposition="outside"))
    fig.update_layout(title=dict(text="Full feature contribution (SHAP)", font=dict(size=13, color="#0f172a")), xaxis=dict(title="SHAP value", gridcolor="#dbe7f3"), yaxis=dict(title=""), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#334155", size=12), margin=dict(t=40, b=20, l=10, r=70))
    return shap_map, fig


def result_csv(patient: dict, probability: float, tier: str, derived: dict) -> bytes:
    """Create a portable one-row assessment summary without storing patient data."""
    row = {
        **patient,
        "PredictedRiskProbability": round(probability, 4),
        "RiskTier": tier,
        "FEV1_FVC_ratio": derived["fev1_fvc"],
        "RespiratorySymptomScore": derived["resp"],
    }
    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def render_aqi_explorer(aqi_df: pd.DataFrame) -> None:
    """Let users explore the bundled AQI data independently of the ML prediction."""
    with st.expander("Explore local AQI context", expanded=False):
        if aqi_df.empty:
            st.info("The optional AQI dataset is not available.")
            return
        states = sorted(aqi_df["state"].dropna().astype(str).unique())
        selected_state = st.selectbox("State", states, key="aqi_state")
        state_data = aqi_df.loc[aqi_df["state"].astype(str) == selected_state].copy()
        areas = sorted(state_data["area"].dropna().astype(str).unique())
        selected_area = st.selectbox("Area", areas, key="aqi_area")
        area_data = state_data.loc[state_data["area"].astype(str) == selected_area].sort_values("date")
        if area_data.empty:
            st.info("No readings are available for this area.")
            return
        latest = area_data.iloc[-1]
        average = area_data["aqi_value"].mean()
        label, color = aqi_label(float(latest["aqi_value"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest AQI", f"{latest['aqi_value']:.0f}", label)
        c2.metric("Average AQI", f"{average:.0f}")
        c3.metric("Peak AQI", f"{area_data['aqi_value'].max():.0f}")
        fig = go.Figure(go.Scatter(
            x=area_data["date"], y=area_data["aqi_value"], mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            title=f"AQI trend — {selected_area}, {selected_state}", height=300,
            xaxis_title="Date", yaxis_title="AQI", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=45, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AQI is shown as environmental context only; it is not included in the current model prediction.")


def build_inputs() -> tuple[dict, bool]:
    with st.form("asthma_input_form"):
        st.markdown("<div class='input-card'><div class='ic-title'>Patient profile</div></div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age (years)", 1, 120, 35, key="age")
        with c2:
            bmi = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1, key="bmi")
        with c3:
            gender = st.selectbox("Gender", ["Female", "Male"], key="gender")
        with c4:
            ethnicity = st.selectbox("Ethnicity (coded)", [0, 1, 2, 3], key="ethnicity")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            fev1 = st.number_input("FEV1 (litres)", 0.5, 8.0, 2.5, step=0.1, key="fev1")
        with c6:
            fvc = st.number_input("FVC (litres)", 0.5, 8.0, 3.2, step=0.1, key="fvc")
        with c7:
            education = st.selectbox("Education level", [0, 1, 2, 3], key="education")
        with c8:
            smoking = yn("Smoking", "smoking", "Has the patient ever smoked?")
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            activity = st.slider("Physical activity (days/week)", 0, 10, 5, key="activity")
        with c10:
            diet = st.slider("Diet quality (0-10)", 0, 10, 6, key="diet")
        with c11:
            sleep = st.slider("Sleep quality (0-10)", 0, 10, 7, key="sleep")
        with c12:
            pollution = st.slider("Pollution exposure (0-10)", 0, 10, 3, key="pollution")
        c13, c14, c15, c16 = st.columns(4)
        with c13:
            pollen = st.slider("Pollen exposure (0-10)", 0, 10, 2, key="pollen")
        with c14:
            dust = st.slider("Dust exposure (0-10)", 0, 10, 3, key="dust")
        with c15:
            pet = yn("Pet allergy", "pet")
        with c16:
            family_history = yn("Family history of asthma", "family_history")
        c17, c18, c19, c20 = st.columns(4)
        with c17:
            allergies = yn("History of allergies", "allergies")
        with c18:
            eczema = yn("Eczema", "eczema")
        with c19:
            hay_fever = yn("Hay fever", "hay_fever")
        with c20:
            reflux = yn("Gastro-oesophageal reflux", "reflux")
        c21, c22, c23, c24 = st.columns(4)
        with c21:
            wheezing = yn("Wheezing", "wheezing")
        with c22:
            shortness = yn("Shortness of breath", "shortness")
        with c23:
            chest_tightness = yn("Chest tightness", "chest_tightness")
        with c24:
            coughing = yn("Coughing", "coughing")
        c25, c26, c27, c28 = st.columns(4)
        with c25:
            nighttime = yn("Nighttime symptoms", "nighttime")
        with c26:
            exercise_induced = yn("Exercise-induced symptoms", "exercise_induced")
        with c27:
            st.write("")
        with c28:
            st.write("")
        if fev1 > fvc:
            st.warning("FEV1 cannot exceed FVC. Please check values.")
        ratio = round(fev1 / fvc, 3) if fvc > 0 else 0
        rcol = "#059669" if ratio >= 0.70 else "#dc2626"
        st.markdown(f"<div style='background:#f8fafc;border:1px solid #dbe7f3;border-radius:10px;padding:10px 16px'><span style='color:#64748b;font-size:12px'>FEV1 / FVC ratio — </span><span style='color:{rcol};font-weight:800;font-size:18px'>{ratio:.2f}</span><span style='color:#64748b;font-size:12px;margin-left:8px'>{('Normal (≥ 0.70)' if ratio >= 0.70 else 'Low — possible obstruction')}</span></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict asthma risk", use_container_width=True)
    patient = {
        "Age": int(age), "BMI": float(bmi), "Gender": 1 if gender == "Male" else 0, "Ethnicity": int(ethnicity), "EducationLevel": int(education), "Smoking": int(smoking), "PhysicalActivity": int(activity), "DietQuality": int(diet), "SleepQuality": int(sleep), "PollutionExposure": int(pollution), "PollenExposure": int(pollen), "DustExposure": int(dust), "PetAllergy": int(pet), "FamilyHistoryAsthma": int(family_history), "HistoryOfAllergies": int(allergies), "Eczema": int(eczema), "HayFever": int(hay_fever), "GastroesophagealReflux": int(reflux), "LungFunctionFEV1": float(fev1), "LungFunctionFVC": float(fvc), "Wheezing": int(wheezing), "ShortnessOfBreath": int(shortness), "ChestTightness": int(chest_tightness), "Coughing": int(coughing), "NighttimeSymptoms": int(nighttime), "ExerciseInduced": int(exercise_induced)
    }
    return patient, submitted


def main() -> None:
    model = load_model()
    preprocessor = load_preprocessor()
    aqi_df = load_aqi()
    report = load_report()
    evaluation_results = load_evaluation_results()
    st.markdown("<div class='hero'><div class='h-title'>Asthma Risk Analysis</div><div class='h-sub'>Explore a patient profile, review a model-estimated risk score, and inspect the factors behind it. This research prototype supports discussion with a qualified clinician; it does not diagnose asthma or replace urgent care.</div><span class='h-chip'>Research prototype</span><span class='h-chip'>No patient data is stored</span></div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi-card'><div class='k-label'>Model Status</div><div class='k-value' style='color:{'#34d399' if model is not None else '#ef4444'}'>{ 'Ready' if model is not None else 'Missing' }</div><div class='k-sub'>Prediction engine</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi-card'><div class='k-label'>Accuracy</div><div class='k-value'>{report.get('Accuracy', '—')}</div><div class='k-sub'>Model performance</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi-card'><div class='k-label'>Positive recall</div><div class='k-value'>{report.get('Positive recall', '—')}</div><div class='k-sub'>Screening reliability</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi-card'><div class='k-label'>ROC-AUC</div><div class='k-value'>{report.get('ROC-AUC', '—')}</div><div class='k-sub'>Discrimination power</div></div>", unsafe_allow_html=True)
    if model is None or preprocessor is None:
        st.error("Model or preprocessor artefacts are missing. Run: python models/train_model.py")
        return
    # Show confusion matrix from evaluation report if available
    if all(k in report for k in ("TN", "FP", "FN", "TP")):
        tn, fp, fn, tp = int(report["TN"]), int(report["FP"]), int(report["FN"]), int(report["TP"])
        total = tn + fp + fn + tp
        st.markdown(f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px">
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:10px 16px;flex:1;min-width:100px;text-align:center">
                <div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase">True Negatives</div>
                <div style="font-size:22px;font-weight:800;color:#166534">{tn}</div>
            </div>
            <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:10px 16px;flex:1;min-width:100px;text-align:center">
                <div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase">False Positives</div>
                <div style="font-size:22px;font-weight:800;color:#991b1b">{fp}</div>
            </div>
            <div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:10px 16px;flex:1;min-width:100px;text-align:center">
                <div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase">False Negatives</div>
                <div style="font-size:22px;font-weight:800;color:#991b1b">{fn}</div>
            </div>
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:10px 16px;flex:1;min-width:100px;text-align:center">
                <div style="font-size:11px;color:#64748b;font-weight:700;text-transform:uppercase">True Positives</div>
                <div style="font-size:22px;font-weight:800;color:#166534">{tp}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    render_research_results(evaluation_results)
    input_toolbar()
    patient, submitted = build_inputs()
    render_aqi_explorer(aqi_df)
    if not submitted:
        st.info("Enter the parameters, then press Predict asthma risk to show the full analysis. You can also load a demonstration profile to explore the interface.")
        return
    if patient["LungFunctionFEV1"] > patient["LungFunctionFVC"]:
        st.error("FEV1 cannot exceed FVC. Correct the lung-function values before running an assessment.")
        return
    patient_df = preprocess_input_data(patient)
    X_scaled = preprocessor.transform(patient_df)
    prob = float(model.predict_proba(X_scaled)[0][1])
    tier = risk_tier(prob)
    derived = compute_derived(patient)
    scaled_df = pd.DataFrame(X_scaled, columns=MODEL_FEATURE_COLUMNS)
    shap_map, shap_fig = shap_full_chart(model, scaled_df)
    rows = build_assessment(patient, derived, shap_map)
    table_html = assessment_table_html(rows)
    st.markdown("<div class='section'><span class='s-num'>1</span><span class='s-title'>Risk Result</span><span class='s-caption'>Predicted outcome & summary</span></div>", unsafe_allow_html=True)
    cfg = RISK_CONFIG[tier]
    st.markdown(f"<div class='risk-banner' style='background:{cfg['bg']};border-color:{cfg['border']};'><div class='rb-title' style='color:{cfg['color']}'>{tier} risk</div><div class='rb-sub' style='color:{cfg['color']}'>{prob * 100:.1f}% predicted probability · {recommendation(tier, derived)}</div></div>", unsafe_allow_html=True)
    st.download_button(
        "Download assessment summary (CSV)",
        data=result_csv(patient, prob, tier, derived),
        file_name="asthma_risk_assessment.csv",
        mime="text/csv",
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.plotly_chart(risk_gauge(prob), use_container_width=True)
        st.markdown("<div class='rec-box'>" + recommendation(tier, derived) + "</div>", unsafe_allow_html=True)
    with right:
        st.plotly_chart(radar_chart(patient, derived), use_container_width=True)
    st.markdown("<div class='section'><span class='s-num'>2</span><span class='s-title'>Input Parameters</span><span class='s-caption'>Healthy range + impact of each factor</span></div>", unsafe_allow_html=True)
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("<div class='section'><span class='s-num'>3</span><span class='s-title'>Clinical Analysis</span><span class='s-caption'>Interpretation and main drivers</span></div>", unsafe_allow_html=True)
    st.markdown(generate_assistant(tier, prob, rows, shap_map), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if shap_fig is not None:
            st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.plotly_chart(drivers_chart(shap_map), use_container_width=True)
    with c2:
        st.info("Interpret feature contributions as model behaviour, not medical causation. AQI context can be explored above and is not included in this prediction.")

if __name__ == "__main__":
    main()
