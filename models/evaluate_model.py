import joblib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             precision_score, average_precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# Ensure the project root is on sys.path when running from the models folder.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from preprocessing import preprocess_data

plt.switch_backend("Agg")


def plot_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run train_model.py first.")
    return joblib.load(model_path)


def main() -> None:
    print("\nLoading preprocessed data for evaluation...\n")
    df = preprocess_data()

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    print("=" * 60)
    print("Class distribution:")
    print(y.value_counts().to_string())
    print(f"Positive class prevalence: {y.mean() * 100:.2f}%")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_path = project_root / "models" / "best_asthma_model.pkl"
    preprocessor_path = project_root / "models" / "preprocessor.pkl"
    metadata_path = project_root / "models" / "model_metadata.json"
    print(f"Loading saved model from {model_path}...\n")
    model = load_model(model_path)
    preprocessor = load_model(preprocessor_path)

    # The champion model was trained on standardized features. Evaluate on the
    # same transformed representation used by the Streamlit application.
    X_test_scaled = preprocessor.transform(X_test)
    scores = model.predict_proba(X_test_scaled)[:, 1]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    threshold = float(metadata.get("decision_threshold", 0.5))
    y_pred = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    precision_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    avg_precision = average_precision_score(y_test, scores)
    roc_auc = roc_auc_score(y_test, scores)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
    print(f"Positive class precision: {precision_pos * 100:.2f}%")
    print(f"Positive class recall: {recall_pos * 100:.2f}%")
    print(f"Positive class F1: {f1_pos * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(f"{'':>12} {'Predicted':>20}")
    print(f"{'':>12} {'Neg':>8} {'Pos':>8}")
    print(f"{'Actual Neg':>12} {tn:>8} {fp:>8}")
    print(f"{'       Pos':>12} {fn:>8} {tp:>8}")
    print()
    print("Classification Report:\n")
    print(report)

    output_dir = project_root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = output_dir / "evaluation_confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, labels=[0, 1], output_path=cm_path)
    print(f"Saved evaluation confusion matrix to {cm_path}")

    report_path = output_dir / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Decision threshold: {threshold:.2f}\n")
        f.write(f"Quality gate passed: {metadata.get('quality_gate_passed', False)}\n")
        f.write(f"Average precision: {avg_precision * 100:.2f}%\n")
        f.write(f"ROC-AUC: {roc_auc * 100:.2f}%\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Positive precision: {precision_pos * 100:.2f}%\n")
        f.write(f"Positive recall: {recall_pos * 100:.2f}%\n")
        f.write(f"Positive F1: {f1_pos * 100:.2f}%\n\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"TN={tn}  FP={fp}\n")
        f.write(f"FN={fn}  TP={tp}\n\n")
        f.write(report)

    print(f"Saved evaluation report to {report_path}")


if __name__ == "__main__":
    main()
