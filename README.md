# Asthma Prediction Project

## Overview

This repository implements an asthma risk prediction pipeline using clinical data from `dataset/asthma.csv`.
The project demonstrates a reproducible machine learning workflow: preprocessing, imbalance handling, model training, model selection, and hold-out evaluation.

## What this project solves

- Predicts asthma risk from patient clinical features
- Handles class imbalance using SMOTE only on the training split
- Trains and compares multiple models
- Selects the best model by positive-class F1 score
- Evaluates the saved model on a held-out 20% test split
- Provides a lightweight Streamlit assessment interface, while keeping the core effort on the ML pipeline

## Repository structure

- `preprocessing.py` — data loading and preprocessing
- `requirements.txt` — required Python packages
- `README.md` — project documentation
- `dataset/`
  - `asthma.csv` — primary dataset for training and evaluation
  - `aqi.csv` — supplemental AQI dataset included for future use
- `models/`
  - `train_model.py` — training pipeline and model selection
  - `evaluate_model.py` — hold-out evaluation of the saved model
  - `best_asthma_model.pkl` — generated after training

## Training pipeline

The training script in `models/train_model.py`:
- loads preprocessed asthma data from `preprocessing.py`
- splits data into train and test sets with `test_size=0.2`, `random_state=42`, and `stratify=y`
- balances the training set using SMOTE
- trains three classification models:
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `XGBoost`
- compares models using accuracy, positive-class recall, and positive-class F1
- selects and saves the best model by positive-class F1 to `models/best_asthma_model.pkl`
- saves evaluation plots for comparison and feature importance

## Evaluation pipeline

The evaluation script in `models/evaluate_model.py`:
- reloads the preprocessing pipeline and data
- reconstructs the same train/test split used during training
- loads `models/best_asthma_model.pkl`
- evaluates the model on the hold-out test set
- prints accuracy, precision, recall, F1, and classification report
- saves a confusion matrix image and a text evaluation report

## Notes on datasets

- `dataset/asthma.csv` is the primary dataset used for training and evaluation.
- `dataset/aqi.csv` is included for future extension and is not currently used by the training/evaluation scripts.

## How to run

From the project root `c:\major_project`:

```powershell
python models\train_model.py
python models\evaluate_model.py
```

If `models/best_asthma_model.pkl` is missing, run `models/train_model.py` first.

