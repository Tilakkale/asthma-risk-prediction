import joblib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Ensure the project root is on sys.path when running from the models folder.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from preprocessing import preprocess_data

plt.switch_backend("Agg")


def plot_accuracy_comparison(metrics: dict[str, float], output_path: Path) -> None:
    model_names = list(metrics.keys())
    accuracies = [metrics[name] * 100 for name in model_names]
    colors = sns.color_palette("viridis", len(model_names))

    plt.figure(figsize=(8, 5))
    plt.barh(model_names, accuracies, color=colors)
    plt.xlabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    for i, value in enumerate(accuracies):
        plt.text(value + 0.5, i, f"{value:.2f}%", va="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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


def plot_feature_importance(feature_names, importances, output_path: Path) -> None:
    importance_df = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    features, scores = zip(*importance_df)
    colors = sns.color_palette("magma", len(features))

    plt.figure(figsize=(10, 6))
    plt.barh(list(features), list(scores), color=colors)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def resample_training_data(X_train, y_train):
    print("\nTraining class distribution before resampling:\n")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("\nTraining class distribution after SMOTE resampling:\n")
    print(y_resampled.value_counts())
    return X_resampled, y_resampled


def main() -> None:
    print("\nLoading Preprocessed Data...\n")
    df = preprocess_data()

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_balanced, y_train_balanced = resample_training_data(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        ),
    }

    model_results = {}
    best_model = None
    best_f1 = 0.0
    best_model_name = ""
    best_predictions = None

    for name, model in models.items():
        print(f"\nTraining {name}...\n")
        model.fit(X_train_balanced, y_train_balanced)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_pos = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        model_results[name] = {
            "accuracy": accuracy,
            "recall_pos": recall_pos,
            "f1_pos": f1_pos,
        }

        print(f"{name} Accuracy: {accuracy * 100:.2f} % | Recall(1): {recall_pos * 100:.2f} % | F1(1): {f1_pos * 100:.2f} %")
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred, zero_division=0))

        if f1_pos > best_f1:
            best_f1 = f1_pos
            best_model = model
            best_model_name = name
            best_predictions = y_pred

    print("\n===================================")
    print(f"\nBest Model by positive F1: {best_model_name}")
    print(f"\nBest positive F1 score: {best_f1 * 100:.2f} %")
    print("\n===================================")

    plot_dir = project_root / "models"
    plot_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = plot_dir / "accuracy_comparison.png"
    plot_accuracy_comparison({name: values['accuracy'] for name, values in model_results.items()}, accuracy_path)
    print(f"\nSaved accuracy comparison chart to {accuracy_path}")

    if best_model is not None and best_predictions is not None:
        cm_path = plot_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_test, best_predictions, labels=[0, 1], output_path=cm_path)
        print(f"Saved confusion matrix to {cm_path}")

        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            importances = abs(best_model.coef_.ravel())
        else:
            perm_result = permutation_importance(
                best_model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
            importances = perm_result.importances_mean

        feature_names = list(X.columns)
        plot_feature_importance(feature_names, importances, plot_dir / "feature_importance.png")
        print(f"Saved feature importance chart to {plot_dir / 'feature_importance.png'}")

    if best_model is not None:
        model_path = plot_dir / "best_asthma_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"\nBest Model Saved Successfully to {model_path}")
    else:
        print("\nNo model was trained successfully.")


if __name__ == "__main__":
    main()
