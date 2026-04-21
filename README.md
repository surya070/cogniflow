# CogniFlow

> Workforce wellbeing intelligence platform — not surveillance, support.

CogniFlow connects Samsung Galaxy Watch biometric data to a personal and team-level health visibility dashboard. Employees see their own sleep quality and energy readiness. Managers see only anonymised, aggregated team signals — never individual scores or biometrics.

---

## Problem Statement
Existing workplace tools do not consider employee physiological state, leading to inefficient workload distribution and increased fatigue. While wearable devices collect relevant biometric data, there is no system that translates this data into actionable insights for workplace use while ensuring privacy.
The problem can be formally stated as: given continuous biometric streams from wearable devices, how can an intelligent system compute a reliable cognitive readiness metric, present it to employees in a meaningful way, and simultaneously preserve individual-level privacy when surfacing team-level insights to managers?

---

## Features by Role

### Employee
- Energy Score (0–100) with colour indicator and score bar
- Today's Insights — rule-engine tips based on last night's data
- Sleep breakdown — deep / REM / light / awake with efficiency %
- Vitals — resting HR, SpO2, HRV, steps
- AI Analysis — Gemini summary with task allocation advice
- History charts — energy and stress trends (7d / 1m / 6m)
- Privacy toggle — opt in to share aggregate status with manager (off by default)

### Manager
- Aggregate team readiness counts (green / amber / red)
- Distribution bar across opted-in employees
- Department grouping when configured
- No individual names, scores, or biometrics visible

### Admin
- User list with role and department assignment

---

## Privacy

| Rule | Enforcement |
|------|-------------|
| No individual biometrics to managers | Manager routes return aggregates only |
| Opt-in sharing, default off | `User.share_readiness = False` |
| Wellness only, not performance | Not connected to HR or performance systems |

---

## ML Pipeline — Dataset, Model, and Energy Score

This is the core technical contribution of the project. The goal was to train a model that predicts a **Samsung-calibrated energy score (0–100)** from sleep and biometric features — without having a large real dataset.

### The Problem

Only one person's real Samsung Galaxy Watch data was available (~5 nights of full-fidelity payloads). A regression model needs hundreds of labelled examples across varied demographics to generalise.

The solution was a three-stage approach:
1. Generate a large, realistic synthetic dataset from physiological first principles
2. Derive a scoring formula by calibrating against real Samsung Health Energy Scores
3. Use that formula to label the synthetic data, then train a supervised model

---

### Stage 1 — Seed Profiles

Before generating any synthetic data, we needed a realistic physiological baseline for each person. We collected one full Samsung-format webhook payload per person from four people, with all fields like their age, sex, etc.

| Person | Age | Sex | Resting HR | Deep% | REM% | Baseline Steps |
|--------|-----|-----|-----------|-------|------|----------------|
| Surya  | 22  | M   | 60 bpm    | 20%   | 22%  | 8 500          |
| Yashu  | 21  | F   | 66 bpm    | 21%   | 24%  | 7 200          |
| Dad    | 51  | M   | 68 bpm    | 14%   | 18%  | 6 000          |
| Mom    | 47  | F   | 72 bpm    | 15%   | 20%  | 5 500          |

The age modulation is intentional — older adults clinically have less deep sleep, higher resting HR, and more fragmented nights. These seeds become the distribution centres for synthetic expansion.

---

### Stage 2 — Synthetic Population (`scripts/populate_dataset.py`)

`populate_dataset.py` expands each seed into **100 synthetic nights** (400 entries total). This is not simple noise addition — every field is sampled from a realistic distribution:

- **Sleep duration** — Gaussian(baseline, σ = 45 min)
- **Sleep start time** — Gaussian(baseline, σ = 30 min), simulating varying bedtimes
- **Stage proportions** — Dirichlet distribution centred on per-person baseline, age-modulated (older → less deep, more awake)
- **Stage sequences** — realistic 90-minute NREM/REM cycles generated minute-by-minute, each stage with a proper start timestamp and duration in seconds — matching the exact Samsung webhook format
- **Heart rate** — per-minute autoregressive AR(1) model producing 1,200+ individual readings per entry; overnight band is lower (60–75 bpm), daytime is higher (70–95 bpm), with age and sex adjustments
- **SpO2** — Gaussian(baseline, σ = 1.0), clipped to [88, 100]
- **Steps** — log-normal daily totals split across realistic daytime time windows

Each of the 400 entries is a **full-fidelity JSON** in exactly the Samsung Health webhook format — `heart_rate` is an array of `{bpm, time}` objects (one per minute), `sleep` contains sessions with `stages` arrays (each with `stage` code, `start_time`, `duration_seconds`), and so on. These are stored in `dataset/entries/`.

The synthetic entries look indistinguishable from real payloads structurally, with the variance engineered to reflect real physiological diversity rather than random noise.

---

### Stage 3 — Feature Extraction (`scripts/extract_features.py`)

Before labelling or training, each full-fidelity JSON is parsed into a flat feature row. This is where the per-minute HR timestamps are actually used — they are **not passed to the model directly**. Instead, `extract_features.py` aggregates them into scalar features:

| Feature | How it's derived |
|---------|-----------------|
| `total_sleep_min` | sum of deep + REM + light stage durations |
| `deep_min`, `rem_min`, `light_min`, `awake_min` | stage code aggregation (`"4"`, `"5"`, `"6"`, `"1"`) |
| `deep_pct`, `rem_pct` | percentage of total sleep |
| `sleep_efficiency_pct` | total_sleep / (total_sleep + awake) × 100 |
| `resting_hr` | 10th percentile of HR readings between 21:00 and 08:00 |
| `avg_hr_overnight` | mean of overnight HR readings |
| `avg_hr_day` | mean of daytime HR readings (08:00–21:00) |
| `hr_std` | standard deviation of all HR readings in the session |
| `hr_min`, `hr_max` | min and max BPM across all readings |
| `spo2_avg`, `spo2_min` | mean and minimum SpO2 % |
| `steps` | sum of all step count windows |
| `age`, `gender`, `height_cm`, `weight_kg` | from `dataset/people.json` |

All 400 extracted rows are written to `dataset/features.csv`. **The model trains on this CSV, not on the raw JSONs.**

The correct Samsung stage code mapping used throughout (not the buggy one in the early processor):
- `"1"` = awake
- `"4"` = deep
- `"5"` = REM
- `"6"` = light

---

### Stage 4 — Label Derivation (`scripts/label_dataset.py`)

This is the key intellectual step. Instead of hand-crafting a scoring formula, we **derived it from real Samsung Health Energy Scores**.

`label_dataset.py` is interactive:

1. Scans `webhook_logs/` for all real watch payloads (one per unique sleep night)
2. Extracts the same features from those real payloads using `extract_features.py`
3. Prompts you to enter the Samsung Health Energy Score shown on your phone for each date
4. Fits a **Ridge regression** (α = n_samples) over five features that are reliable on real Samsung data:

| Feature | Why these five |
|---------|---------------|
| `total_sleep_min` | always present in real payloads |
| `sleep_efficiency_pct` | derived from session start/end — always reliable |
| `resting_hr` | 10th percentile of overnight HR |
| `spo2_avg` | average of SpO2 readings |
| `steps` | always present |

`deep_pct` and `rem_pct` are deliberately excluded. The real Samsung Galaxy Watch only sends stage codes `"1"` (awake) and `"4"` (deep) — REM and light are not reported in real payloads, making those percentages unreliable for formula derivation.

5. Prints a fit table (predicted vs. actual Samsung scores, RMSE, R²)
6. Saves `dataset/derived_formula.json` — the coefficients and intercept
7. Applies the formula to all 400 synthetic entries, clips to [0, 100], writes `energy_score` back into `features.csv`

**The four real Samsung scores used for calibration:**

| Date | Samsung Energy Score |
|------|---------------------|
| 2026-03-18 | 80 |
| 2026-03-19 | 76 |
| 2026-04-19 | 85 |
| 2026-04-20 | 66 |

Ridge regularisation (α = 4) stabilises coefficients with this small sample. Primary drivers are `total_sleep_min` and `sleep_efficiency_pct`.

---

### Stage 5 — Model Training (`scripts/train_model.py`)

`train_model.py` trains two models side-by-side on the 400 labelled rows from `features.csv`:

- **LinearRegression** — baseline; because the labels are a linear formula, this fits near-perfectly
- **RandomForestRegressor(n_estimators=200)** — ensemble; matches the linear model closely and provides feature importances

Both are evaluated on an 80/20 held-out split (RMSE, MAE, R²). The better model by RMSE is saved to `models/energy_score_model.joblib` alongside a feature importance chart at `models/feature_importance.png`.

The joblib file stores the model object, the ordered feature name list, the model type string, and the test metrics — everything needed to make predictions and audit the model later.

---

### How the Energy Score Works in the App

When a webhook arrives from the watch:

1. `services/processor.py` extracts all features from the raw payload
2. `services/energy_score.py` loads `dataset/derived_formula.json` and applies: `score = intercept + Σ(coef × feature)`, clipped to [0, 100]
3. If the formula file is missing, falls back to the rule engine percentage
4. The score is stored in `watch_readings.energy_score` and used throughout the dashboard

Score bands:

| Score | Colour | Meaning |
|-------|--------|---------|
| 70–100 | Green | Well recovered |
| 40–69 | Amber | Moderate recovery |
| 0–39 | Red | Low energy |

---

### Running the Pipeline

```bash
venv/Scripts/python scripts/build_seeds.py          # create seed profiles (one-time)
venv/Scripts/python scripts/populate_dataset.py     # generate 400 synthetic entries
venv/Scripts/python scripts/label_dataset.py        # INTERACTIVE — enter real Samsung scores
venv/Scripts/python scripts/train_model.py          # train and save model
```

---

## Webhook Integration

Each user gets a personal webhook URL on their Profile page:

```
POST http://<your-server>/webhook/watch?token=<personal_token>
Content-Type: application/json

{ "sleep": [...], "heart_rate": [...], "oxygen_saturation": [...], "steps": [...] }
```

Configure Health Connect on Android with an automation app (HTTP Shortcuts, Tasker, or Health Auto Export) to POST daily after waking. The server returns `200` immediately; Gemini analysis runs in the background.

---

## Google OAuth Setup (fixing "Permission denied to generate login hint")

1. Go to **Google Cloud Console → APIs & Services → Credentials**
2. Open your OAuth 2.0 Client ID
3. **Authorised JavaScript origins** → add `http://localhost:5000`
4. **Authorised redirect URIs** → add `http://localhost:5000/auth/google/callback`
5. Go to **OAuth consent screen → Test users** → confirm your email is listed
6. Make sure **User type** is **External** (Internal requires a Google Workspace org)
7. Save and wait ~5 minutes

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file:

```
SECRET_KEY=your-secret-key
GOOGLE_API_KEY=your-gemini-api-key

# Optional — only needed for Google OAuth login
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
```

```bash
python app.py
# → http://localhost:5000
```

The database is created automatically on first run.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python 3.11+) |
| Database | SQLite via SQLAlchemy |
| Auth | Flask-Login + email/password + Google OAuth |
| ML pipeline | scikit-learn, pandas, numpy, joblib |
| LLM analysis | Google Gemini API |
| Frontend | Jinja2, Chart.js |
| Watch integration | Samsung Galaxy Watch → Health Connect → webhook POST |

---
