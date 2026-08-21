"""Train and evaluate candidates without selecting on the final test set.

Run from the project root:
    python models/train_model.py
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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from preprocessing import preprocess_data

MODEL_PATH = PROJECT_ROOT / "models" / "best_asthma_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_report.txt"


def metrics_at_threshold(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "predicted_positive_count": int(predictions.sum()),
    }


def choose_threshold(y_validation: pd.Series, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    """Pick threshold using Youden's J statistic (sensitivity + specificity - 1)
    with a minimum threshold floor of 0.15 to avoid trivial solutions."""
    candidates = np.unique(np.r_[0.01, np.arange(0.02, 1.00, 0.01), 0.99])
    best_j = -1.0
    best_t = 0.5
    best_metrics = metrics_at_threshold(y_validation, scores, 0.5)
    for t in candidates:
        if t < 0.15:
            continue  # Skip trivial near-zero thresholds
        m = metrics_at_threshold(y_validation, scores, float(t))
        sens = m["recall"]
        spec = accuracy_score(y_validation, (scores < t).astype(int))  # quick specificity proxy
        # Proper specificity calculation
        y_pred = (scores >= t).astype(int)
        tn = ((y_validation == 0) & (y_pred == 0)).sum()
        fp = ((y_validation == 0) & (y_pred == 1)).sum()
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec_val - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


def fit_candidate(name: str, X_train: pd.DataFrame, y_train: pd.Series):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    # Apply SMOTE to ALL models to handle severe class imbalance (~5% positives)
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_scaled, y_train)
    if name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    elif name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        )
    else:
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=float((y_train == 0).sum() / y_train.sum()),
            eval_metric="logloss", random_state=42, n_jobs=1,
        )
    model.fit(X_resampled, y_resampled)
    return scaler, model


def plot_confusion(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    cm = confusion_matrix(y_true, scores >= threshold, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Hold-out confusion matrix")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "models" / "evaluation_confusion_matrix.png", dpi=140)
    plt.close()
    return {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])}


def main() -> None:
    data = preprocess_data()
    X, y = data.drop(columns="Diagnosis"), data["Diagnosis"]

    print("=" * 60)
    print("Class distribution in full dataset:")
    print(y.value_counts().to_string())
    print(f"Positive class prevalence: {y.mean() * 100:.2f}%")
    print("=" * 60)

    # Keep this final 20% split untouched until the final evaluation.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=42,
    )

    print(f"\nTraining set size: {len(X_train)} (positives: {y_train.sum()})")
    print(f"Validation set size: {len(X_validation)} (positives: {y_validation.sum()})")
    print(f"Test set size: {len(X_test)} (positives: {y_test.sum()})")

    candidates = ("Logistic Regression", "Random Forest", "XGBoost")
    selection: list[dict[str, object]] = []
    for name in candidates:
        scaler, model = fit_candidate(name, X_train, y_train)
        validation_scores = model.predict_proba(scaler.transform(X_validation))[:, 1]
        threshold, operating_point = choose_threshold(y_validation, validation_scores)
        selection.append({
            "name": name,
            "average_precision": float(average_precision_score(y_validation, validation_scores)),
            "roc_auc": float(roc_auc_score(y_validation, validation_scores)),
            "threshold": threshold,
            "operating_point": operating_point,
        })

    # Average precision evaluates ranking quality without pretending a threshold is clinical.
    champion = max(selection, key=lambda item: float(item["average_precision"]))
    name = str(champion["name"])
    threshold = float(champion["threshold"])
    scaler, model = fit_candidate(name, X_trainval, y_trainval)
    test_scores = model.predict_proba(scaler.transform(X_test))[:, 1]
    test_metrics = metrics_at_threshold(y_test, test_scores, threshold)
    test_ap = float(average_precision_score(y_test, test_scores))
    test_roc_auc = float(roc_auc_score(y_test, test_scores))
    baseline_ap = float(y_validation.mean())

    # A model whose validation AP does not beat prevalence has no evidence of useful ranking signal.
    quality_gate_passed = bool(float(champion["average_precision"]) >= baseline_ap * 1.10)
    metadata = {
        "model_name": name,
        "decision_threshold": threshold,
        "selection_metric": "validation_average_precision",
        "quality_gate_passed": quality_gate_passed,
        "quality_gate_reason": (
            "Validation average precision must be at least 10% above positive-class prevalence."
        ),
        "validation": {
            "positive_prevalence": baseline_ap,
            "average_precision": champion["average_precision"],
            "roc_auc": champion["roc_auc"],
            "threshold_metrics": champion["operating_point"],
        },
        "test": {
            "positive_support": int(y_test.sum()),
            "average_precision": test_ap,
            "roc_auc": test_roc_auc,
            **test_metrics,
        },
        "candidate_selection": selection,
    }
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, PREPROCESSOR_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_lines = [
        f"Model: {name}",
        f"Decision threshold (validation selected): {threshold:.2f}",
        f"Quality gate passed: {quality_gate_passed}",
        f"Validation average precision: {champion['average_precision'] * 100:.2f}%",
        f"Validation positive prevalence: {baseline_ap * 100:.2f}%",
        f"Test average precision: {test_ap * 100:.2f}%",
        f"Test ROC-AUC: {test_roc_auc * 100:.2f}%",
        f"Accuracy: {test_metrics['accuracy'] * 100:.2f}%",
        f"Positive precision: {test_metrics['precision'] * 100:.2f}%",
        f"Positive recall: {test_metrics['recall'] * 100:.2f}%",
        f"Positive F1: {test_metrics['f1'] * 100:.2f}%",
        f"Positive F2: {test_metrics['f2'] * 100:.2f}%",
        f"Predicted positives: {test_metrics['predicted_positive_count']} / {len(y_test)}",
    ]
    cm_values = plot_confusion(y_test, test_scores, threshold)
    report_lines.append(f"Confusion matrix: TP={cm_values['TP']}  FN={cm_values['FN']}  FP={cm_values['FP']}  TN={cm_values['TN']}")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Neg   Pos")
    print(f"Actual Neg    {cm_values['TN']:>4}  {cm_values['FP']:>4}")
    print(f"       Pos    {cm_values['FN']:>4}  {cm_values['TP']:>4}")
    print(f"\nSaved model metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
