# CogniFlow

> Workforce wellbeing intelligence platform — not surveillance, support.

CogniFlow connects Samsung Galaxy Watch biometric data to a team-level health visibility dashboard. Employees get a personal sleep and recovery dashboard. Managers see only anonymised, aggregated team health signals — never individual data.

---

## How the Dataset Was Built

### The Problem: No Multi-Person Data
The only real watch data available was from one person (Surya) — roughly 5 nights of full-fidelity payloads from a Samsung Galaxy Watch via Health Connect. A regression model needs hundreds of labelled examples across multiple people with different age/gender profiles.

### Step 1 — Seed Profiles
`scripts/build_seeds.py` creates one baseline payload per person using age/gender-tuned physiological parameters:

| Person | Age | Sex | Resting HR | Deep% | REM% | Baseline Steps |
|--------|-----|-----|-----------|-------|------|----------------|
| Surya  | 22  | M   | 60 bpm    | 20%   | 22%  | 8 500          |
| Yashu  | 21  | F   | 66 bpm    | 21%   | 24%  | 7 200          |
| Dad    | 51  | M   | 68 bpm    | 14%   | 18%  | 6 000          |
| Mom    | 47  | F   | 72 bpm    | 15%   | 20%  | 5 500          |


### Step 2 — Synthetic Population (`populate_dataset.py`) ⭐
`scripts/populate_dataset.py` expands each seed into 100 synthetic nights (400 total). Variance is applied to **every field** — nothing is left fixed:

- **Sleep duration** — Gaussian(baseline, σ = 45 min)
- **Sleep start time** — Gaussian(baseline, σ = 30 min)
- **Stage proportions** — Dirichlet distribution around per-person baseline, age-modulated (older → less deep sleep)
- **Stage sequences** — realistic 90-minute cycles generated minute-by-minute with proper start/end timestamps
- **Heart rate** — per-minute autoregressive model (1 200+ readings per entry), overnight band lower, daytime higher, age/gender adjusted
- **SpO2** — Gaussian(baseline, σ = 1), clipped [88, 100]
- **Steps** — log-normal daily totals split across time windows

Each of the 400 entries is a **full-fidelity JSON** matching the exact Samsung Health webhook format — HR per minute, stages with timestamps, SpO2 per reading. Nothing is aggregated.

Stage codes used: `"1"` awake, `"4"` deep, `"5"` REM, `"6"` light.

### Step 3 — Formula Derivation (`label_dataset.py`) ⭐
`scripts/label_dataset.py` is an **interactive** script that derives the energy score formula from real Samsung data:

1. Walks `webhook_logs/` and finds all real full-fidelity payloads, one per unique sleep night.
2. Prompts the user for the Samsung Health Energy Score shown on his phone for each of those dates.
3. Extracts features from those real payloads.
4. Fits a **Ridge regression** (`alpha = n_samples`) over the five most reliable features:

   | Feature | Why reliable on real data |
   |---------|--------------------------|
   | `total_sleep_min` | Always present |
   | `sleep_efficiency_pct` | Derived from session start/end timestamps |
   | `resting_hr` | 10th percentile of overnight HR readings |
   | `spo2_avg` | Average of SpO2 readings |
   | `steps` | Always present |

   Deep% and REM% are deliberately excluded: the real Samsung Galaxy Watch only sends stage codes `"1"` (awake) and `"4"` (deep), so those percentages are unreliable on real data.

5. Prints a fit table showing predicted vs. actual Samsung scores plus RMSE and R².
6. Saves `dataset/derived_formula.json` with coefficients and intercept.
7. Applies the formula to all 400 synthetic entries, clips to [0, 100], writes `energy_score` back into `dataset/features.csv`.

This is the key intellectual step: the formula is **derived from real Samsung scores**, not a hand-crafted heuristic.

### Step 4 — Model Training (`train_model.py`)
`scripts/train_model.py` trains two models side-by-side on the now-labelled 400-entry dataset:

- `LinearRegression` — baseline
- `RandomForestRegressor(n_estimators=200)` — ensemble

Features include age, gender (one-hot), and all extracted sleep/HR/SpO2/steps columns. The best model (by RMSE on an 80/20 held-out split) is saved to `models/energy_score_model.joblib`.

Because the labels are a deterministic linear formula, linear regression fits near-perfectly and the forest matches it closely. The value of this step is having a deployable predictor and a feature-importance chart (`models/feature_importance.png`).

### Pipeline

```
venv/Scripts/python scripts/build_seeds.py          # one-time
venv/Scripts/python scripts/populate_dataset.py     # generates 400 entries
venv/Scripts/python scripts/label_dataset.py        # INTERACTIVE — ask user for real scores
venv/Scripts/python scripts/train_model.py          # train + save model
```

---

## How the Energy Score Is Used in the App

When a new webhook arrives from the watch, `app.py` runs the following pipeline:

1. **Processor** — extracts sleep stages (correct mapping: `"4"` = deep, `"5"` = REM, `"6"` = light), resting HR (10th percentile of overnight readings), SpO2, steps, sleep efficiency.
2. **Energy Score** — the derived formula is applied directly: `score = intercept + Σ(coef × feature)`, clipped to [0, 100]. Falls back to rule engine percentage if the formula file is missing.
3. **Stress Score** — derived from resting HR, sleep duration, and HRV (when available). Independent of the energy score.
4. **Rule Engine** — categorical labels (high / moderate / low) for backward compatibility.
5. **Gemini LLM** — runs in a background thread (returns 200 immediately to the watch app, preventing timeout).

Score bands:

| Score | Colour | Meaning |
|-------|--------|---------|
| 70–100 | Green | Well recovered |
| 40–69 | Amber | Moderate |
| 0–39 | Red | Low energy / underrecovered |

---

## Use Case

CogniFlow is a **workforce wellbeing intelligence** tool. The core insight:

> "Given last night's sleep and recovery data, what cognitive state is this person in today, and how should their workday be structured?"

### Employee Portal
- **Energy Score (0–100)** — primary readiness number, Samsung-calibrated.
- **Stress Score (0–100)** — derived from resting HR, sleep duration, HRV.
- **Sleep Breakdown** — deep / REM / light / awake bars with sleep efficiency %.
- **Vitals** — resting HR, SpO2, HRV, steps.
- **AI Analysis** — Gemini summary with key insights and task allocation advice.
- **History Charts** — energy and stress trends over 7 days / 1 month / 6 months.
- **Privacy toggle** — share aggregate status with the team dashboard (off by default).

### Manager Portal
- Sees only employees who opted in to sharing.
- Aggregate green / amber / red counts and a distribution bar.
- Department grouping when configured by admin.
- Never shows individual names, scores, or biometrics.

### Admin Portal
- User list with role assignment (employee / manager / admin).
- Department assignment for team grouping.
- Score threshold reference.

---

## Privacy Guarantees

| Rule | Enforcement |
|------|-------------|
| No individual biometrics visible to managers | Manager routes return aggregates only |
| Employee opt-in sharing, default off | `User.share_readiness = False` by default |
| Wellness only, not performance | Not connected to HR or performance systems |
| Data stays within the organisation | No third-party data sharing |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python 3.11+) |
| Database | SQLite via SQLAlchemy |
| Auth | Flask-Login + email/password + Google OAuth (optional) |
| ML pipeline | scikit-learn, pandas, numpy, joblib |
| LLM analysis | Google Gemini API |
| Frontend | Jinja2 templates, Chart.js, custom dark-theme CSS |
| Watch integration | Samsung Galaxy Watch → Health Connect → webhook POST |

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# Create a .env file with:
#   SECRET_KEY=your-secret-key
#   GOOGLE_API_KEY=your-gemini-api-key

# 4. Run the app
python app.py
# → http://localhost:5000

# 5. (Optional) Run the ML pipeline if not already done
python scripts/build_seeds.py
python scripts/populate_dataset.py
python scripts/label_dataset.py   # interactive
python scripts/train_model.py
```

The database is created and migrated automatically on first run. The derived formula (`dataset/derived_formula.json`) is loaded at startup for real-time energy score computation.

---

## Webhook Integration

Each user gets a personal webhook URL on their Profile page:

```
POST http://<your-server>/webhook/watch?token=<personal_token>
Content-Type: application/json

{ "sleep": [...], "heart_rate": [...], "oxygen_saturation": [...], "steps": [...] }
```

Configure Health Connect on Android + an automation app (HTTP Shortcuts, Tasker, or Health Auto Export) to POST daily after waking up. The server returns `200` immediately; Gemini analysis runs in the background.

---

## Score Calibration

The energy score formula was calibrated on 4 real Samsung Health Energy Score readings:

| Date | Samsung Score |
|------|--------------|
| 2026-03-18 | 80 |
| 2026-03-19 | 76 |
| 2026-04-19 | 85 |
| 2026-04-20 | 66 |

Ridge regularisation (α = 4) stabilises coefficients with this small sample. Primary drivers are `total_sleep_min` and `sleep_efficiency_pct`. Resting HR and SpO2 flow through the stress score independently.
