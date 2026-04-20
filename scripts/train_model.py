"""
Train a supervised regressor that predicts energy_score from the extracted
features in dataset/features.csv.

Must be run AFTER scripts/label_dataset.py has filled the energy_score column.

Fits two models side-by-side (LinearRegression baseline + RandomForestRegressor),
reports RMSE / MAE / R^2 on a held-out split, saves the better one to
models/energy_score_model.joblib, and saves a feature-importance chart.

Run:
    venv/Scripts/python scripts/train_model.py
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET_DIR   = os.path.join(ROOT, "dataset")
FEATURES_CSV  = os.path.join(DATASET_DIR, "features.csv")
MODELS_DIR    = os.path.join(ROOT, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "energy_score_model.joblib")
IMPORTANCE_PNG = os.path.join(MODELS_DIR, "feature_importance.png")

DROP_COLS = {"entry_id", "person_id", "sleep_date", "energy_score"}


def _prepare_xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    y = df["energy_score"].astype(float).values

    # One-hot encode gender (M/F -> 2 cols); other categorical cols stay numeric.
    if "gender" in X.columns:
        X = pd.get_dummies(X, columns=["gender"], drop_first=False)

    # Any remaining non-numeric column becomes a zero (shouldn't happen).
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.fillna(0.0)
    return X, y, list(X.columns)


def _report(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:<20}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R^2={r2:6.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: {FEATURES_CSV} not found. Run populate_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    if "energy_score" not in df.columns or df["energy_score"].isna().all() \
            or (df["energy_score"].astype(str) == "").all():
        print("ERROR: energy_score column is empty. Run label_dataset.py first.")
        sys.exit(1)

    # Drop rows with missing labels (shouldn't happen after labeling, but safe).
    df = df[pd.to_numeric(df["energy_score"], errors="coerce").notna()].copy()
    df["energy_score"] = df["energy_score"].astype(float)
    print(f"Loaded {len(df)} labeled rows.")
    print(f"energy_score: min={df['energy_score'].min():.1f} "
          f"mean={df['energy_score'].mean():.1f} "
          f"max={df['energy_score'].max():.1f} "
          f"std={df['energy_score'].std():.2f}")

    X, y, feat_names = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train / test split: {len(X_train)} / {len(X_test)}")

    print()
    print("Training ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print()
    print("Test metrics:")
    lr_metrics = _report("LinearRegression", y_test, lr.predict(X_test))
    rf_metrics = _report("RandomForest",     y_test, rf.predict(X_test))

    # Pick the better model (by RMSE)
    if rf_metrics["rmse"] <= lr_metrics["rmse"]:
        best, best_name, best_metrics = rf, "RandomForestRegressor", rf_metrics
    else:
        best, best_name, best_metrics = lr, "LinearRegression", lr_metrics

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        "model":        best,
        "feature_names": feat_names,
        "model_type":   best_name,
        "metrics":      best_metrics,
    }, MODEL_PATH)
    print()
    print(f"Saved best model ({best_name}) -> {os.path.relpath(MODEL_PATH, ROOT)}")

    # Feature importance chart (RF only)
    try:
        importances = rf.feature_importances_
        order = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(importances)), importances[order])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feat_names[i] for i in order], rotation=45, ha="right")
        ax.set_ylabel("Importance")
        ax.set_title("Random Forest feature importances")
        fig.tight_layout()
        fig.savefig(IMPORTANCE_PNG, dpi=120)
        plt.close(fig)
        print(f"Saved importance chart -> {os.path.relpath(IMPORTANCE_PNG, ROOT)}")
    except Exception as exc:
        print(f"(skipped importance chart: {exc})")


if __name__ == "__main__":
    main()
