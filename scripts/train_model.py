"""
Train, compare, tune, and save the best energy score predictor.

Steps:
  1. Train 7 models on 80/20 split — compare RMSE / MAE / R2
  2. Save every model to models/
  3. Save comparison table to models/evaluation_results.csv
  4. Hyperparameter-tune the best model (GridSearchCV, 5-fold CV)
  5. Save tuned model to models/energy_score_model.joblib (the production model)
  6. Save before/after tuning comparison to models/tuning_comparison.csv
  7. Save feature importance chart

Run:
    venv/Scripts/python scripts/train_model.py
"""
import os
import sys
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)
FEATURES_CSV = os.path.join(ROOT, "dataset", "features.csv")
MODELS_DIR   = os.path.join(ROOT, "models")

DROP_COLS = {"entry_id", "person_id", "sleep_date", "energy_score"}


# ── helpers ───────────────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    y = df["energy_score"].astype(float).values
    if "gender" in X.columns:
        X = pd.get_dummies(X, columns=["gender"], drop_first=False)
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X, y, list(X.columns)


def metrics(name, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    print(f"  {name:<30}  RMSE={rmse:6.3f}  MAE={mae:6.3f}  R2={r2:7.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def separator(title=""):
    print()
    print("-" * 65)
    if title:
        print(f"  {title}")
        print("-" * 65)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(FEATURES_CSV):
        print(f"ERROR: {FEATURES_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    df = df[pd.to_numeric(df["energy_score"], errors="coerce").notna()].copy()
    df["energy_score"] = df["energy_score"].astype(float)
    print(f"Loaded {len(df)} labelled rows.")
    print(f"energy_score  min={df['energy_score'].min():.1f}  "
          f"mean={df['energy_score'].mean():.1f}  "
          f"max={df['energy_score'].max():.1f}  "
          f"std={df['energy_score'].std():.2f}")

    X, y, feat_names = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train / test split: {len(X_train)} / {len(X_test)}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Stage 1: train all candidates ────────────────────────────────────────
    separator("STAGE 1 — Training all candidate models")

    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=4.0),
        "RandomForest":     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "ExtraTrees":       ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "XGBoost":          xgb.XGBRegressor(n_estimators=200, random_state=42,
                                              verbosity=0, n_jobs=-1),
        "LightGBM":         lgb.LGBMRegressor(n_estimators=200, random_state=42,
                                               verbose=-1, n_jobs=-1),
        # SVR needs scaling — wrap in pipeline
        "SVR":              Pipeline([("scaler", StandardScaler()),
                                      ("svr",    SVR(kernel="rbf", C=10, epsilon=0.5))]),
    }

    results = []
    trained = {}

    print()
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = metrics(name, y_test, y_pred)

        # 5-fold CV RMSE on the full dataset for a more stable estimate
        cv_scores = cross_val_score(model, X, y, cv=5,
                                    scoring="neg_root_mean_squared_error", n_jobs=-1)
        m["cv_rmse"] = float(-cv_scores.mean())
        m["cv_rmse_std"] = float(cv_scores.std())

        results.append(m)
        trained[name] = model

        # Save individual model
        save_path = os.path.join(MODELS_DIR, f"{name.lower()}.joblib")
        joblib.dump({"model": model, "feature_names": feat_names,
                     "model_type": name, "metrics": m}, save_path)

    # Save evaluation table
    eval_df = pd.DataFrame(results).sort_values("rmse")
    eval_csv = os.path.join(MODELS_DIR, "evaluation_results.csv")
    eval_df.to_csv(eval_csv, index=False)

    separator("STAGE 1 — Results (sorted by test RMSE)")
    print()
    print(eval_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  Saved -> {os.path.relpath(eval_csv, ROOT)}")

    # Pick best by test RMSE
    best_name = eval_df.iloc[0]["model"]
    best_pre  = eval_df.iloc[0].to_dict()
    print(f"\n  Best model: {best_name}  (RMSE={best_pre['rmse']:.4f}  R2={best_pre['r2']:.4f})")

    # ── Stage 2: hyperparameter tuning on the best model ─────────────────────
    separator(f"STAGE 2 — Hyperparameter tuning: {best_name}")

    param_grids = {
        "LinearRegression": {},   # no hyperparameters
        "Ridge": {
            "alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        },
        "RandomForest": {
            "n_estimators":      [100, 200, 400],
            "max_depth":         [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.7],
        },
        "ExtraTrees": {
            "n_estimators":      [100, 200, 400],
            "max_depth":         [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.7],
        },
        "GradientBoosting": {
            "n_estimators":  [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth":     [2, 3, 5],
            "subsample":     [0.7, 0.85, 1.0],
        },
        "XGBoost": {
            "n_estimators":  [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
        "LightGBM": {
            "n_estimators":  [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth":     [-1, 5, 10],
            "num_leaves":    [15, 31, 63],
            "subsample":     [0.7, 0.85, 1.0],
        },
        "SVR": {
            "svr__C":       [0.1, 1, 10, 50, 100],
            "svr__epsilon": [0.1, 0.5, 1.0, 2.0],
            "svr__gamma":   ["scale", "auto"],
        },
    }

    grid = param_grids.get(best_name, {})

    if not grid:
        print(f"  No hyperparameters to tune for {best_name} — keeping as-is.")
        tuned_model = trained[best_name]
        best_post = best_pre.copy()
    else:
        base_model = candidates[best_name]  # fresh unfitted instance

        search = GridSearchCV(
            base_model,
            grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        print(f"  Running GridSearchCV (5-fold) over {best_name}...")
        search.fit(X_train, y_train)

        tuned_model = search.best_estimator_
        print(f"\n  Best params: {search.best_params_}")
        print(f"  Best CV RMSE (train): {-search.best_score_:.4f}")

        y_pred_tuned = tuned_model.predict(X_test)
        best_post = metrics(f"{best_name} (tuned)", y_test, y_pred_tuned)
        best_post["model"] = f"{best_name} (tuned)"

        cv_tuned = cross_val_score(tuned_model, X, y, cv=5,
                                   scoring="neg_root_mean_squared_error", n_jobs=-1)
        best_post["cv_rmse"]     = float(-cv_tuned.mean())
        best_post["cv_rmse_std"] = float(cv_tuned.std())

    # ── Stage 3: before/after comparison ─────────────────────────────────────
    separator("STAGE 3 — Before vs After Tuning")
    print()

    comparison = pd.DataFrame([best_pre, best_post])
    comparison_csv = os.path.join(MODELS_DIR, "tuning_comparison.csv")
    comparison.to_csv(comparison_csv, index=False)

    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    rmse_delta = best_pre["rmse"] - best_post["rmse"]
    r2_delta   = best_post["r2"]  - best_pre["r2"]
    print(f"\n  RMSE improvement : {rmse_delta:+.4f}  ({'better' if rmse_delta > 0 else 'no change'})")
    print(f"  R2   improvement : {r2_delta:+.4f}  ({'better' if r2_delta  > 0 else 'no change'})")
    print(f"\n  Saved -> {os.path.relpath(comparison_csv, ROOT)}")

    # ── Stage 4: save the production model ───────────────────────────────────
    separator("STAGE 4 — Saving production model")

    prod_path = os.path.join(MODELS_DIR, "energy_score_model.joblib")
    joblib.dump({
        "model":         tuned_model,
        "feature_names": feat_names,
        "model_type":    best_name,
        "metrics":       best_post,
        "tuned":         bool(grid),
    }, prod_path)
    print(f"\n  Saved production model -> {os.path.relpath(prod_path, ROOT)}")

    # ── Stage 5: feature importance chart ────────────────────────────────────
    try:
        # works for tree-based models; SVR/linear don't have feature_importances_
        inner = tuned_model
        if hasattr(inner, "named_steps"):
            inner = inner.named_steps.get(list(inner.named_steps)[-1], inner)
        importances = inner.feature_importances_
        order = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(importances)), importances[order],
               color="#5c6bc0", edgecolor="white")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feat_names[i] for i in order], rotation=45, ha="right")
        ax.set_ylabel("Importance")
        ax.set_title(f"Feature importances — {best_name} (tuned)")
        fig.tight_layout()
        chart_path = os.path.join(MODELS_DIR, "feature_importance.png")
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        print(f"  Saved importance chart -> {os.path.relpath(chart_path, ROOT)}")
    except AttributeError:
        print("  (feature importance chart skipped — model has no feature_importances_)")

    # ── Final summary ─────────────────────────────────────────────────────────
    separator("FINAL SUMMARY")
    print()
    print(f"  Best model        : {best_name}")
    print(f"  Tuned             : {bool(grid)}")
    print(f"  Test RMSE (final) : {best_post['rmse']:.4f}")
    print(f"  Test MAE  (final) : {best_post['mae']:.4f}")
    print(f"  Test R2   (final) : {best_post['r2']:.4f}")
    print(f"  CV RMSE   (final) : {best_post['cv_rmse']:.4f} ± {best_post['cv_rmse_std']:.4f}")
    print()
    print("  Output files:")
    for fname in sorted(os.listdir(MODELS_DIR)):
        print(f"    models/{fname}")
    print()


if __name__ == "__main__":
    main()
