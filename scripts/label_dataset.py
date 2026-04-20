"""
INTERACTIVE labeler.

Scans webhook_logs/ for Surya's real full-fidelity payloads, prompts the user
for the Samsung Energy Score that his phone showed for each of those nights,
fits a formula from those (features -> score) pairs using Ridge regression,
prints the fit quality, then applies the formula to label every row in
dataset/features.csv.

This is intentionally one script because the whole flow is one conceptual step:
'label the dataset using a formula derived from real Samsung scores'.

Run:
    venv/Scripts/python scripts/label_dataset.py
"""
import json
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_features import extract_features

ROOT          = os.path.dirname(HERE)
WEBHOOK_LOGS  = os.path.join(ROOT, "webhook_logs")
DATASET_DIR   = os.path.join(ROOT, "dataset")
FEATURES_CSV  = os.path.join(DATASET_DIR, "features.csv")
FORMULA_JSON  = os.path.join(DATASET_DIR, "derived_formula.json")

# Features the formula is expressed over. These are chosen to be reliable on
# BOTH the real data (which only uses stage codes 1 and 4 on Samsung Galaxy
# Watches — see memory/project_data.md) AND the synthetic data (which uses all
# 4 stage codes). deep_pct/rem_pct are deliberately omitted because they're
# unreliable on real data.
FORMULA_FEATURES = [
    "total_sleep_min",
    "sleep_efficiency_pct",
    "resting_hr",
    "spo2_avg",
    "steps",
]


def _find_real_payloads() -> list[tuple[str, str, dict]]:
    """
    Walk webhook_logs/ for full-fidelity real payloads.

    Returns list of (sleep_date, path, payload) — one entry per UNIQUE sleep
    session end date. If the same night appears in multiple dumps, the largest
    file wins.
    """
    by_date: dict[str, tuple[int, str, dict]] = {}

    for root, _dirs, files in os.walk(WEBHOOK_LOGS):
        for name in files:
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if not payload.get("sleep") or not payload.get("heart_rate"):
                continue

            # A single dump can contain multiple sleep sessions — split them.
            for session in payload["sleep"]:
                session_end = session.get("session_end_time")
                if not session_end:
                    continue
                date_str = session_end[:10]

                # Rebuild a single-session payload for feature extraction
                single = dict(payload)
                single["sleep"] = [session]

                size = os.path.getsize(path)
                existing = by_date.get(date_str)
                if existing is None or size > existing[0]:
                    by_date[date_str] = (size, path, single)

    out = []
    for date_str in sorted(by_date.keys()):
        _size, path, payload = by_date[date_str]
        out.append((date_str, path, payload))
    return out


def _prompt_scores(reals: list[tuple[str, str, dict]]) -> OrderedDict:
    """Prompt user for Samsung Energy Score on each real night."""
    print()
    print("=" * 70)
    print("  Samsung Energy Score capture")
    print("=" * 70)
    print("For each night below, open your Samsung Health app and enter the")
    print("Energy Score it showed. Enter a number 0-100, or press Enter / 'skip'")
    print("to skip a night. You need at least 3 scores to fit a stable formula.")
    print()

    scores = OrderedDict()
    for date_str, path, _payload in reals:
        rel = os.path.relpath(path, ROOT)
        while True:
            raw = input(f"  Energy Score for {date_str}  (source: {rel}): ").strip()
            if raw == "" or raw.lower() == "skip":
                print("    skipped")
                break
            try:
                val = float(raw)
            except ValueError:
                print("    not a number, try again")
                continue
            if not (0 <= val <= 100):
                print("    out of range 0-100, try again")
                continue
            scores[date_str] = val
            break
    return scores


def _features_matrix(real_payloads: list[tuple[str, str, dict]],
                     scores: OrderedDict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X, y arrays aligned with the user-provided scores."""
    rows = []
    y = []
    dates = []
    for date_str, _path, payload in real_payloads:
        if date_str not in scores:
            continue
        feats = extract_features(payload)
        x = []
        for col in FORMULA_FEATURES:
            v = feats.get(col)
            if v is None:
                v = 0.0
            x.append(float(v))
        rows.append(x)
        y.append(scores[date_str])
        dates.append(date_str)
    return np.array(rows), np.array(y), dates


def _fit_and_save(X: np.ndarray, y: np.ndarray, dates: list[str]) -> dict:
    """Fit Ridge, print side-by-side, save weights to derived_formula.json."""
    n_samples = len(y)
    # Use moderate L2 (dataset is tiny → strong regularization helps)
    alpha = max(1.0, float(n_samples))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)

    y_hat = model.predict(X)
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size > 1 else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    print()
    print("-" * 70)
    print("  Fit quality")
    print("-" * 70)
    print(f"  {'Date':<12} {'Samsung':>8} {'Predicted':>10} {'Diff':>7}")
    for d, truth, pred in zip(dates, y, y_hat):
        print(f"  {d:<12} {truth:>8.1f} {pred:>10.1f} {truth - pred:>+7.1f}")
    print(f"  {'RMSE':<12} {rmse:>8.2f}")
    print(f"  {'R^2':<12} {r2:>8.3f}")
    print()

    print("  Formula:  energy_score = intercept + sum(coef[i] * feature[i])")
    print(f"    intercept = {float(model.intercept_):+.4f}")
    for feat, coef in zip(FORMULA_FEATURES, model.coef_):
        print(f"    {feat:<24} {float(coef):+.6f}")
    print()

    formula = {
        "features":            FORMULA_FEATURES,
        "coefficients":        [float(c) for c in model.coef_],
        "intercept":           float(model.intercept_),
        "ridge_alpha":         alpha,
        "n_training_samples":  int(n_samples),
        "r2_on_training":      float(r2) if not np.isnan(r2) else None,
        "rmse_on_training":    rmse,
        "training_dates":      dates,
        "training_scores":     [float(v) for v in y],
    }
    with open(FORMULA_JSON, "w", encoding="utf-8") as f:
        json.dump(formula, f, indent=2)
    print(f"  Saved formula -> {os.path.relpath(FORMULA_JSON, ROOT)}")
    return formula


def _apply_to_csv(formula: dict):
    df = pd.read_csv(FEATURES_CSV)
    X = df[FORMULA_FEATURES].fillna(0.0).values
    coefs = np.array(formula["coefficients"])
    intercept = formula["intercept"]
    scores = X @ coefs + intercept
    scores = np.clip(scores, 0.0, 100.0)
    df["energy_score"] = np.round(scores, 2)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"  Labeled {len(df)} rows in {os.path.relpath(FEATURES_CSV, ROOT)}")
    print(f"  energy_score distribution: min={df['energy_score'].min():.1f} "
          f"mean={df['energy_score'].mean():.1f} max={df['energy_score'].max():.1f}")


def main():
    reals = _find_real_payloads()
    if not reals:
        print("No real full-fidelity payloads found in webhook_logs/.")
        sys.exit(1)

    print(f"Found {len(reals)} real night(s) of watch data:")
    for date_str, path, _ in reals:
        rel = os.path.relpath(path, ROOT)
        print(f"  - {date_str}  ({rel})")

    scores = _prompt_scores(reals)
    if len(scores) < 3:
        print(f"\nERROR: Need at least 3 scores to fit; got {len(scores)}.")
        print("       Re-run and enter more scores.")
        sys.exit(1)

    X, y, dates = _features_matrix(reals, scores)
    formula = _fit_and_save(X, y, dates)
    _apply_to_csv(formula)


if __name__ == "__main__":
    main()
