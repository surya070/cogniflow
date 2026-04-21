# Model Selection — CogniFlow Energy Score Predictor

## Task

Regression: predict a continuous energy score (0–100) from 24 sleep and biometric features extracted from Samsung Galaxy Watch data. The score is a linear combination of five features (derived by Ridge regression against real Samsung Health scores), so the ground-truth labels are smooth and near-linear — but with demographic and per-night variance introduced across 400 synthetic entries.

---

## Models Evaluated and Why Each Was Chosen

### Linear Regression
The simplest possible baseline. Since the labels themselves come from a linear formula, this should perform reasonably well. Included to establish a floor and to verify that tree models genuinely add value beyond the label structure.

### Ridge Regression
Linear regression with L2 regularisation. The label derivation used Ridge (alpha=4), so this is the closest structural match to how the labels were generated. Expected to perform similarly to LinearRegression on this dataset. Useful as a sanity check: if Ridge doesn't beat LinearRegression by much, the regularisation penalty wasn't doing heavy lifting.

### Random Forest
The model that was previously in production. A bagged ensemble of decision trees — robust to outliers, handles non-linearities, and provides feature importances. Chosen because it is the industry standard baseline for tabular regression and because it was already validated in the previous pipeline.

### Extra Trees (Extremely Randomised Trees)
Similar to Random Forest but splits are chosen randomly rather than optimally at each node. Often faster to train and can generalise better when the dataset is small. Included as a direct comparison to Random Forest to see if the extra randomisation helps on 400 rows.

### Gradient Boosting (sklearn)
A sequential ensemble that builds trees one at a time, each correcting the residuals of the last. Generally achieves lower bias than bagged ensembles on structured/tabular data. Chosen because it consistently outperforms Random Forest on small-to-medium tabular datasets where variance is low and bias reduction matters more.

### XGBoost
A highly optimised implementation of gradient boosting with built-in regularisation, column subsampling, and shrinkage. Included because it is the most widely used gradient boosting library and often beats sklearn's GradientBoosting on speed and regularisation quality.

### LightGBM
Gradient boosting with leaf-wise tree growth (vs. depth-wise in XGBoost). Grows deeper trees faster and often achieves lower error on datasets with many features. Included to compare against XGBoost — both are strong candidates and the winner is dataset-dependent.

### SVR (Support Vector Regression)
Finds a hyperplane that fits the most points within an epsilon margin, ignoring outliers beyond that band. Wrapped in a StandardScaler pipeline because SVR is sensitive to feature scale. Included as a non-tree baseline to test whether the structure of this problem (near-linear labels, moderate variance) benefits from margin-based regression.

---

## Stage 1 — Model Comparison (80/20 train/test split + 5-fold CV)

All models trained on the same 320-row training set, evaluated on the same 80-row held-out test set. CV RMSE is the mean across 5 folds on the full 400-row dataset.

| Model | Test RMSE | Test MAE | Test R² | CV RMSE | CV RMSE ± |
|---|---|---|---|---|---|
| **GradientBoosting** | **0.5444** | **0.4236** | **0.9963** | **0.5859** | 0.0487 |
| LightGBM | 0.6633 | 0.4597 | 0.9944 | 0.7061 | 0.0879 |
| ExtraTrees | 0.6900 | 0.5588 | 0.9940 | 0.6849 | 0.0572 |
| RandomForest | 0.7642 | 0.5880 | 0.9926 | 0.8348 | 0.0300 |
| XGBoost | 0.7815 | 0.5902 | 0.9923 | 0.8220 | 0.0732 |
| SVR | 1.6261 | 1.2021 | 0.9666 | 1.8223 | 0.3197 |
| Ridge | 1.6581 | 1.0474 | 0.9653 | 1.8199 | 0.4409 |
| LinearRegression | 1.6815 | 1.0482 | 0.9643 | 1.8273 | 0.4370 |

### Accuracy at tolerance thresholds

"Within N points" = percentage of test predictions within N of the true score. Band accuracy = percentage where the predicted band (high/moderate/low) matches the true band.

| Model | Within 1pt | Within 2pt | Within 5pt | Band Accuracy |
|---|---|---|---|---|
| **GradientBoosting** | **93.8%** | **100.0%** | **100.0%** | **100.0%** |
| LightGBM | 90.0% | 97.5% | 100.0% | 98.8% |
| ExtraTrees | 88.8% | 100.0% | 100.0% | 98.8% |
| RandomForest | 83.8% | 98.8% | 100.0% | 100.0% |
| XGBoost | 78.8% | 98.8% | 100.0% | 100.0% |
| SVR | 56.2% | 81.2% | 98.8% | 97.5% |
| Ridge | 65.0% | 90.0% | 97.5% | 98.8% |
| LinearRegression | 65.0% | 90.0% | 97.5% | 98.8% |

### Why GradientBoosting won

- Best test RMSE (0.5444) and best CV RMSE (0.5859) — consistent across both evaluations
- Only model with 100% within-2-points accuracy before tuning
- Perfect band classification accuracy — every prediction lands in the correct high/moderate/low tier
- Low CV standard deviation (±0.049) indicates stable generalisation, not a lucky split
- LightGBM was close (RMSE 0.663) but GradientBoosting's lower variance made it safer to tune
- Linear models were outperformed because the 400-entry dataset has age/gender interactions and demographic non-linearities that a linear formula can't fully capture, even though the labels are near-linear

---

## Stage 2 — Hyperparameter Tuning (GradientBoosting)

GridSearchCV with 5-fold cross-validation. 108 parameter combinations × 5 folds = 540 fits.

**Search space:**

| Parameter | Values searched |
|---|---|
| `n_estimators` | 100, 200, 400 |
| `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| `max_depth` | 2, 3, 5 |
| `subsample` | 0.7, 0.85, 1.0 |

**Best parameters found:**

```
learning_rate  = 0.05
max_depth      = 3
n_estimators   = 400
subsample      = 0.7
```

Why these make sense: a lower learning rate (0.05 vs default 0.1) with more trees (400) is the classic boosting sweet spot — each tree corrects less aggressively, reducing overfitting. Shallow trees (max_depth=3) prevent individual trees from memorising noise. Subsampling 70% of training rows per tree (subsample=0.7) adds stochasticity and further reduces variance.

---

## Stage 3 — Before vs After Tuning

| | Test RMSE | Test MAE | Test R² | CV RMSE | CV RMSE ± |
|---|---|---|---|---|---|
| GradientBoosting (default) | 0.5444 | 0.4236 | 0.9963 | 0.5859 | 0.0487 |
| GradientBoosting (tuned) | **0.4684** | **0.3538** | **0.9972** | **0.5082** | 0.0315 |
| Improvement | +0.0760 RMSE | +0.0698 MAE | +0.0010 R² | +0.0777 CV RMSE | |

### Accuracy at tolerance thresholds — before vs after

| | Within 1pt | Within 2pt | Within 5pt | Band Accuracy |
|---|---|---|---|---|
| GradientBoosting (default) | 93.8% | 100.0% | 100.0% | 100.0% |
| GradientBoosting (tuned) | **97.5%** | **100.0%** | **100.0%** | **100.0%** |
| Improvement | +3.7% | — | — | — |

Tuning improved within-1-point accuracy from 93.8% to 97.5% — meaning 78 out of 80 test predictions are within 1 point of the true score. The model predicts the correct readiness band (high/moderate/low) for every single test example.

---

## Final Production Model

**GradientBoosting (tuned)**

```
Test RMSE  : 0.4684
Test MAE   : 0.3538
Test R²    : 0.9972
CV RMSE    : 0.5082 +/- 0.0315
Within 1pt : 97.5%
Within 2pt : 100.0%
Band Acc   : 100.0%
```

Saved to `models/energy_score_model.joblib`.

---

## Output Files

| File | Contents |
|---|---|
| `energy_score_model.joblib` | Production model (tuned GradientBoosting) |
| `evaluation_results.csv` | Stage 1 metrics for all 8 models |
| `tuning_comparison.csv` | Before/after tuning metrics |
| `feature_importance.png` | Feature importance chart (tuned model) |
| `gradientboosting.joblib` | GradientBoosting (default params) |
| `lightgbm.joblib` | LightGBM |
| `extratrees.joblib` | Extra Trees |
| `randomforest.joblib` | Random Forest (previously in production) |
| `xgboost.joblib` | XGBoost |
| `svr.joblib` | SVR |
| `ridge.joblib` | Ridge Regression |
| `linearregression.joblib` | Linear Regression |
