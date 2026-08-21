"""
Comprehensive model evaluation — run from project root:
    python models/comprehensive_evaluation.py

Produces:
  - models/evaluation_results.json   (all metrics)
  - models/evaluation_*.png           (all plots)
  - Console output with full analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from preprocessing import preprocess_data, MODEL_FEATURE_COLUMNS

OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
print("=" * 70)
print("COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)

df = preprocess_data()
X = df.drop(columns="Diagnosis")
y = df["Diagnosis"]

print(f"\nDataset: {len(df)} samples, {len(X.columns)} features")
print(f"Positive class: {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Negative class: {(1-y).sum()} ({(1-y).mean()*100:.2f}%)")

# Load saved model + preprocessor
model_path = PROJECT_ROOT / "models" / "best_asthma_model.pkl"
prep_path = PROJECT_ROOT / "models" / "preprocessor.pkl"
meta_path = PROJECT_ROOT / "models" / "model_metadata.json"

if not model_path.exists():
    print("\nERROR: No saved model found. Run python models/train_model.py first.")
    sys.exit(1)

model = joblib.load(model_path)
preprocessor = joblib.load(prep_path)
metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
threshold = float(metadata.get("decision_threshold", 0.5))
model_name = metadata.get("model_name", "Unknown")

print(f"\nLoaded model: {model_name}")
print(f"Decision threshold: {threshold:.3f}")

# ──────────────────────────────────────────────
# 2. Train/test split (same as train_model.py)
# ──────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=42,
)

X_test_scaled = preprocessor.transform(X_test)
scores = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (scores >= threshold).astype(int)

results: dict[str, object] = {}

# ──────────────────────────────────────────────
# 3. Majority-class baseline
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("MAJORITY-CLASS BASELINE")
print("─" * 70)

y_majority = np.zeros_like(y_test)
majority_acc = accuracy_score(y_test, y_majority)
majority_prec = precision_score(y_test, y_majority, zero_division=0)
majority_rec = recall_score(y_test, y_majority, zero_division=0)
majority_f1 = f1_score(y_test, y_majority, zero_division=0)
print(f"Majority-class (always predict 0):")
print(f"  Accuracy:  {majority_acc*100:.2f}%")
print(f"  Precision: {majority_prec*100:.2f}%")
print(f"  Recall:    {majority_rec*100:.2f}%")
print(f"  F1-score:  {majority_f1*100:.2f}%")

results["majority_baseline"] = {
    "accuracy": majority_acc,
    "precision": majority_prec,
    "recall": majority_rec,
    "f1": majority_f1,
}

# ──────────────────────────────────────────────
# 4. Model metrics at threshold
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"MODEL PERFORMANCE (threshold = {threshold:.3f})")
print("─" * 70)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"  Accuracy:  {acc*100:.2f}%")
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall:    {rec*100:.2f}%")
print(f"  F1-score:  {f1*100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"                Predicted")
print(f"                Neg   Pos")
print(f"  Actual Neg    {tn:>4}  {fp:>4}")
print(f"         Pos    {fn:>4}  {tp:>4}")

results["model"] = {
    "name": model_name,
    "threshold": threshold,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
}

# ──────────────────────────────────────────────
# 5. ROC-AUC + PR-AUC
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("RANKING METRICS (threshold-independent)")
print("─" * 70)

roc_auc = roc_auc_score(y_test, scores)
pr_auc = average_precision_score(y_test, scores)
print(f"  ROC-AUC: {roc_auc*100:.2f}%")
print(f"  PR-AUC:  {pr_auc*100:.2f}%")

results["ranking_metrics"] = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
}

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, scores)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"ROC-AUC = {roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random (0.50)")
ax.fill_between(fpr, tpr, alpha=0.10, color="#2563eb")
ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
plt.tight_layout()
roc_path = OUTPUT_DIR / "evaluation_roc_curve.png"
plt.savefig(roc_path, dpi=140)
plt.close()
print(f"  Saved ROC curve → {roc_path.name}")

# PR curve plot
precision_vals, recall_vals, _ = precision_recall_curve(y_test, scores)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_vals, precision_vals, color="#059669", lw=2.5, label=f"PR-AUC = {pr_auc:.3f}")
baseline_pr = y_test.mean()
ax.axhline(y=baseline_pr, color="gray", linestyle="--", lw=1.5, alpha=0.5, label=f"Baseline ({baseline_pr:.3f})")
ax.fill_between(recall_vals, precision_vals, alpha=0.10, color="#059669")
ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax.set_ylabel("Precision (PPV)", fontsize=12)
ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(alpha=0.25)
plt.tight_layout()
pr_path = OUTPUT_DIR / "evaluation_pr_curve.png"
plt.savefig(pr_path, dpi=140)
plt.close()
print(f"  Saved PR curve   → {pr_path.name}")

# ──────────────────────────────────────────────
# 6. Confusion matrix plot
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
            ax=ax, annot_kws={"fontsize": 16, "fontweight": "bold"})
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})", fontsize=13, fontweight="bold")
plt.tight_layout()
cm_path = OUTPUT_DIR / "evaluation_confusion_matrix.png"
plt.savefig(cm_path, dpi=140)
plt.close()
print(f"  Saved CM plot    → {cm_path.name}")

# ──────────────────────────────────────────────
# 7. Threshold analysis
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("THRESHOLD ANALYSIS")
print("─" * 70)

thresholds = np.arange(0.05, 0.96, 0.02)
thresh_results = []
for t in thresholds:
    p = (scores >= t).astype(int)
    thresh_results.append({
        "threshold": float(t),
        "accuracy": float(accuracy_score(y_test, p)),
        "precision": float(precision_score(y_test, p, zero_division=0)),
        "recall": float(recall_score(y_test, p, zero_division=0)),
        "f1": float(f1_score(y_test, p, zero_division=0)),
        "predicted_positives": int(p.sum()),
    })

# Find best F1 threshold
best_f1_t = max(thresh_results, key=lambda r: r["f1"])
print(f"  Best F1 threshold:     {best_f1_t['threshold']:.2f} (F1 = {best_f1_t['f1']:.3f})")
print(f"  Current threshold:     {threshold:.2f} (F1 = {f1:.3f})")
print(f"  Majority baseline F1:  {majority_f1:.3f}")

results["threshold_analysis"] = thresh_results

# Threshold plot
df_thresh = pd.DataFrame(thresh_results)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(df_thresh["threshold"], df_thresh["precision"], label="Precision", color="#2563eb", lw=2)
ax1.plot(df_thresh["threshold"], df_thresh["recall"], label="Recall", color="#059669", lw=2)
ax1.plot(df_thresh["threshold"], df_thresh["f1"], label="F1-score", color="#d97706", lw=2.5)
ax1.axvline(x=threshold, color="red", linestyle="--", alpha=0.5, label=f"Current threshold ({threshold:.2f})")
ax1.set_xlabel("Decision Threshold", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title("Threshold vs Precision / Recall / F1", fontsize=14, fontweight="bold")
ax1.legend(loc="center right", fontsize=11)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.02)
ax1.grid(alpha=0.25)

ax2 = ax1.twinx()
ax2.plot(df_thresh["threshold"], df_thresh["predicted_positives"], color="#7c3aed", lw=1.5, linestyle="--", alpha=0.6, label="Predicted positives")
ax2.set_ylabel("Predicted Positives", fontsize=12, color="#7c3aed")
ax2.tick_params(axis="y", labelcolor="#7c3aed")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
plt.tight_layout()
thresh_path = OUTPUT_DIR / "evaluation_threshold_analysis.png"
plt.savefig(thresh_path, dpi=140)
plt.close()
print(f"  Saved threshold plot → {thresh_path.name}")

# ──────────────────────────────────────────────
# 8. Calibration analysis
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("CALIBRATION ANALYSIS")
print("─" * 70)

brier = brier_score_loss(y_test, scores)
print(f"  Brier score: {brier:.4f} (0 = perfect, >0.25 = uninformative)")

prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=10, strategy="uniform")
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, "o-", color="#2563eb", lw=2.5, markersize=8, label="Model")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Perfect calibration")
ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.10, color="#2563eb")
ax.set_xlabel("Mean Predicted Probability", fontsize=12)
ax.set_ylabel("Observed Fraction of Positives", fontsize=12)
ax.set_title(f"Calibration Curve (Brier = {brier:.4f})", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
plt.tight_layout()
cal_path = OUTPUT_DIR / "evaluation_calibration_curve.png"
plt.savefig(cal_path, dpi=140)
plt.close()
print(f"  Saved calibration plot → {cal_path.name}")

results["calibration"] = {
    "brier_score": brier,
}

# ──────────────────────────────────────────────
# 9. Repeated stratified k-fold CV
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("REPEATED STRATIFIED K-FOLD CROSS-VALIDATION")
print("─" * 70)

# Use a pipeline approach for CV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

cv_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1,
    )),
])

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_auc = cross_val_score(cv_pipeline, X_trainval, y_trainval, cv=rskf, scoring="roc_auc", n_jobs=-1)
cv_ap = cross_val_score(cv_pipeline, X_trainval, y_trainval, cv=rskf, scoring="average_precision", n_jobs=-1)

print(f"  ROC-AUC:       {cv_auc.mean()*100:.2f}% ± {cv_auc.std()*100:.2f}%")
print(f"  PR-AUC (AP):   {cv_ap.mean()*100:.2f}% ± {cv_ap.std()*100:.2f}%")

results["repeated_cv"] = {
    "roc_auc_mean": float(cv_auc.mean()),
    "roc_auc_std": float(cv_auc.std()),
    "pr_auc_mean": float(cv_ap.mean()),
    "pr_auc_std": float(cv_ap.std()),
}

# ──────────────────────────────────────────────
# 10. Feature-group ablation
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("FEATURE-GROUP ABLATION")
print("─" * 70)

feature_groups = {
    "Full (all features)": MODEL_FEATURE_COLUMNS,
    "Clinical only": ["Age", "Gender", "Ethnicity", "EducationLevel", "BMI"],
    "Clinical + Lifestyle": ["Age", "Gender", "Ethnicity", "EducationLevel", "BMI",
                              "Smoking", "PhysicalActivity", "DietQuality", "SleepQuality"],
    "Without respiratory": [c for c in MODEL_FEATURE_COLUMNS if c not in (
        "Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing",
        "NighttimeSymptoms", "ExerciseInduced", "RespiratorySymptomScore")],
    "Without medical/allergy": [c for c in MODEL_FEATURE_COLUMNS if c not in (
        "FamilyHistoryAsthma", "HistoryOfAllergies", "Eczema", "HayFever",
        "GastroesophagealReflux", "HighRiskHistory", "PetAllergy",
        "AllergyExposureScore", "PollenExposure", "DustExposure")],
    "Lung function only": ["LungFunctionFEV1", "LungFunctionFVC", "FEV1_FVC_ratio"],
    "Symptoms only": ["Wheezing", "ShortnessOfBreath", "ChestTightness", "Coughing",
                       "NighttimeSymptoms", "ExerciseInduced", "RespiratorySymptomScore"],
}

ablation_results = {}
for group_name, cols in feature_groups.items():
    available = [c for c in cols if c in X_trainval.columns]
    if len(available) < 3:
        continue
    X_subset = X_trainval[available]
    try:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                random_state=42, n_jobs=-1,
            )),
        ])
        cv_auc_sub = cross_val_score(pipe, X_subset, y_trainval, cv=5, scoring="roc_auc", n_jobs=-1)
        cv_ap_sub = cross_val_score(pipe, X_subset, y_trainval, cv=5, scoring="average_precision", n_jobs=-1)
        ablation_results[group_name] = {
            "n_features": len(available),
            "roc_auc_mean": float(cv_auc_sub.mean()),
            "roc_auc_std": float(cv_auc_sub.std()),
            "pr_auc_mean": float(cv_ap_sub.mean()),
            "pr_auc_std": float(cv_ap_sub.std()),
        }
        print(f"  {group_name:35s}  ROC-AUC: {cv_auc_sub.mean()*100:.1f}% ± {cv_auc_sub.std()*100:.1f}%  |  PR-AUC: {cv_ap_sub.mean()*100:.1f}% ± {cv_ap_sub.std()*100:.1f}%  (n={len(available)})")
    except Exception as e:
        print(f"  {group_name:35s}  ERROR: {e}")

results["feature_ablation"] = ablation_results

# Ablation plot
if ablation_results:
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(ablation_results.keys())
    rocs = [ablation_results[n]["roc_auc_mean"] * 100 for n in names]
    aps = [ablation_results[n]["pr_auc_mean"] * 100 for n in names]
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, rocs, w, label="ROC-AUC", color="#2563eb", alpha=0.8)
    bars2 = ax.bar(x + w/2, aps, w, label="PR-AUC", color="#059669", alpha=0.8)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Feature-Group Ablation (5-fold CV)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    abl_path = OUTPUT_DIR / "evaluation_feature_ablation.png"
    plt.savefig(abl_path, dpi=140)
    plt.close()
    print(f"  Saved ablation plot → {abl_path.name}")

# ──────────────────────────────────────────────
# 11. Error analysis
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("ERROR ANALYSIS")
print("─" * 70)

error_df = X_test.copy()
error_df["true_label"] = y_test.values
error_df["predicted"] = y_pred
error_df["probability"] = scores
error_df["error_type"] = "Correct"
error_df.loc[(error_df["true_label"] == 1) & (error_df["predicted"] == 1), "error_type"] = "TP"
error_df.loc[(error_df["true_label"] == 0) & (error_df["predicted"] == 0), "error_type"] = "TN"
error_df.loc[(error_df["true_label"] == 0) & (error_df["predicted"] == 1), "error_type"] = "FP"
error_df.loc[(error_df["true_label"] == 1) & (error_df["predicted"] == 0), "error_type"] = "FN"

print(f"  True Positives:  {tp}")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")

# Feature means by error type
error_means = error_df.groupby("error_type")[MODEL_FEATURE_COLUMNS].mean()
print("\n  Feature means by error type (top discriminative features):")
for col in MODEL_FEATURE_COLUMNS[:8]:
    vals = error_means[col] if col in error_means.columns else None
    if vals is not None:
        print(f"    {col:25s}  TP={vals.get('TP', 0):.2f}  FN={vals.get('FN', 0):.2f}  FP={vals.get('FP', 0):.2f}  TN={vals.get('TN', 0):.2f}")

results["error_analysis"] = {
    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
}

# ──────────────────────────────────────────────
# 12. Subgroup analysis
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("SUBGROUP ANALYSIS")
print("─" * 70)

subgroup_results = {}
subgroups = {
    "Age < 30": X_test["Age"] < 30,
    "Age 30–50": (X_test["Age"] >= 30) & (X_test["Age"] <= 50),
    "Age > 50": X_test["Age"] > 50,
    "Female": X_test["Gender"] == 0,
    "Male": X_test["Gender"] == 1,
    "BMI < 25": X_test["BMI"] < 25,
    "BMI 25–30": (X_test["BMI"] >= 25) & (X_test["BMI"] <= 30),
    "BMI > 30": X_test["BMI"] > 30,
}

for sg_name, sg_mask in subgroups.items():
    sg_y = y_test.values[sg_mask.values]
    sg_scores = scores[sg_mask.values]
    if len(sg_y) < 5 or len(np.unique(sg_y)) < 2:
        continue
    sg_auc = roc_auc_score(sg_y, sg_scores)
    sg_ap = average_precision_score(sg_y, sg_scores)
    sg_pred = (sg_scores >= threshold).astype(int)
    sg_acc = accuracy_score(sg_y, sg_pred)
    sg_rec = recall_score(sg_y, sg_pred, zero_division=0)
    sg_prec = precision_score(sg_y, sg_pred, zero_division=0)
    subgroup_results[sg_name] = {
        "n": int(len(sg_y)),
        "n_positives": int(sg_y.sum()),
        "roc_auc": sg_auc,
        "pr_auc": sg_ap,
        "accuracy": sg_acc,
        "recall": sg_rec,
        "precision": sg_prec,
    }
    print(f"  {sg_name:15s}  n={len(sg_y):4d}  pos={sg_y.sum():3d}  ROC-AUC={sg_auc*100:.1f}%  PR-AUC={sg_ap*100:.1f}%  Acc={sg_acc*100:.1f}%  Rec={sg_rec*100:.1f}%  Prec={sg_prec*100:.1f}%")

results["subgroup_analysis"] = subgroup_results

# ──────────────────────────────────────────────
# 13. SHAP analysis (global + local)
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("SHAP ANALYSIS")
print("─" * 70)

try:
    import shap
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=MODEL_FEATURE_COLUMNS)
    
    # Determine explainer type
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_type = "TreeExplainer"
    else:
        explainer = shap.LinearExplainer(model, X_test_scaled_df)
        shap_type = "LinearExplainer"
    
    shap_values = explainer.shap_values(X_test_scaled_df)
    # shap_values shape: (n_samples, n_features, n_classes) for multi-output
    if isinstance(shap_values, list):
        shap_values_class = shap_values[1]  # positive class
    elif shap_values.ndim == 3:
        shap_values_class = shap_values[:, :, 1]  # positive class
    else:
        shap_values_class = shap_values
    
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray) and ev.ndim > 0:
        ev_pos = float(ev[1]) if len(ev) > 1 else float(ev[0])
    elif isinstance(ev, list):
        ev_pos = float(ev[1]) if len(ev) > 1 else float(ev[0])
    else:
        ev_pos = float(ev)
    
    # Global SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values_class, X_test_scaled_df, show=False, max_display=15, alpha=0.6)
    plt.title(f"Global SHAP Feature Importance ({shap_type})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    shap_global_path = OUTPUT_DIR / "evaluation_shap_global.png"
    plt.savefig(shap_global_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP global plot → {shap_global_path.name}")
    
    # Local SHAP waterfall for first test positive
    pos_indices = np.where(y_test.values == 1)[0]
    if len(pos_indices) > 0:
        idx = pos_indices[0]
        shap_val_row = shap_values_class[idx]
        data_row = X_test_scaled_df.iloc[idx].values
        try:
            fig = plt.figure(figsize=(10, 7))
            shap.waterfall_plot(
                shap.Explanation(values=shap_val_row,
                               base_values=ev_pos,
                               data=data_row,
                               feature_names=MODEL_FEATURE_COLUMNS),
                max_display=12, show=False,
            )
            plt.title(f"Local SHAP Explanation (Patient #{idx}, True Positive)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            shap_local_path = OUTPUT_DIR / "evaluation_shap_local.png"
            plt.savefig(shap_local_path, dpi=140, bbox_inches="tight")
            plt.close()
            print(f"  Saved SHAP local plot  → {shap_local_path.name}")
        except Exception as e:
            print(f"  SHAP waterfall plot failed (fallback to bar): {e}")
            # Fallback: bar plot of top features
            fig, ax = plt.subplots(figsize=(10, 6))
            indices = np.argsort(np.abs(shap_val_row))[-12:]
            colors = ["#ef4444" if v > 0 else "#059669" for v in shap_val_row[indices]]
            ax.barh(range(len(indices)), shap_val_row[indices], color=colors)
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([MODEL_FEATURE_COLUMNS[i] for i in indices], fontsize=10)
            ax.axvline(x=0, color="gray", linestyle="-", lw=0.5)
            ax.set_xlabel("SHAP Value", fontsize=12)
            ax.set_title(f"Local SHAP (Patient #{idx}, True Positive)", fontsize=13, fontweight="bold")
            plt.tight_layout()
            shap_local_path = OUTPUT_DIR / "evaluation_shap_local.png"
            plt.savefig(shap_local_path, dpi=140, bbox_inches="tight")
            plt.close()
            print(f"  Saved SHAP local bar plot → {shap_local_path.name}")
    
    results["shap"] = {
        "explainer_type": shap_type,
        "global_plot": str(shap_global_path.name),
        "local_plot": str(shap_local_path.name) if len(pos_indices) > 0 else None,
    }
except ImportError:
    print("  SHAP not installed. Install with: pip install shap")
    results["shap"] = {"error": "SHAP not installed"}
except Exception as e:
    print(f"  SHAP analysis failed: {e}")
    results["shap"] = {"error": str(e)}

# ──────────────────────────────────────────────
# 14. Validation status
# ──────────────────────────────────────────────
print("\n" + "─" * 70)
print("VALIDATION STATUS")
print("─" * 70)

quality_passed = metadata.get("quality_gate_passed", False)
validation_status = "NOT VALIDATED"
if quality_passed:
    validation_status = "VALIDATED (quality gate passed)"
elif roc_auc > 0.55:
    validation_status = "PARTIALLY VALIDATED (ROC-AUC > 0.55 but quality gate not met)"
else:
    validation_status = "NOT VALIDATED (ROC-AUC near random)"

print(f"  Quality gate passed: {quality_passed}")
print(f"  ROC-AUC: {roc_auc*100:.2f}%")
print(f"  Status: {validation_status}")

results["validation_status"] = {
    "quality_gate_passed": quality_passed,
    "roc_auc": roc_auc,
    "status": validation_status,
}

# ──────────────────────────────────────────────
# 15. Save all results
# ──────────────────────────────────────────────
results_path = OUTPUT_DIR / "evaluation_results.json"
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Saved all results → {results_path.name}")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)