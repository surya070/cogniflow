"""
Shared feature extractor for the ML pipeline.

Input:  a full Health Auto Export-style payload dict (with sleep, heart_rate,
        oxygen_saturation, steps arrays) + optional person metadata.
Output: a flat feature dict ready for a CSV row.

Uses the CORRECT stage-code mapping (1=awake, 4=deep, 5=rem, 6=light).
Does NOT reuse services/processor.py because of its known stage-code bug.
"""
from datetime import datetime, timezone


STAGE_AWAKE = {"1"}
STAGE_DEEP  = {"4"}
STAGE_REM   = {"5"}
STAGE_LIGHT = {"6"}


def _parse_dt(value):
    if not value:
        return None
    value = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def extract_features(payload: dict, person_meta: dict | None = None,
                     entry_id: str | None = None,
                     person_id: str | None = None) -> dict:
    # ── Sleep stage aggregation ─────────────────────────────────────────
    deep_s = rem_s = light_s = awake_s = 0
    sleep_start_dt = None
    sleep_end_dt   = None

    for session in (payload.get("sleep") or []):
        s_end = _parse_dt(session.get("session_end_time"))
        if s_end and (sleep_end_dt is None or s_end > sleep_end_dt):
            sleep_end_dt = s_end

        for stg in (session.get("stages") or []):
            code = str(stg.get("stage", "")).strip()
            dur  = int(stg.get("duration_seconds") or 0)
            if   code in STAGE_DEEP:  deep_s  += dur
            elif code in STAGE_REM:   rem_s   += dur
            elif code in STAGE_LIGHT: light_s += dur
            elif code in STAGE_AWAKE: awake_s += dur

            s_start = _parse_dt(stg.get("start_time"))
            if s_start and (sleep_start_dt is None or s_start < sleep_start_dt):
                sleep_start_dt = s_start

    deep_min  = deep_s  / 60
    rem_min   = rem_s   / 60
    light_min = light_s / 60
    awake_min = awake_s / 60
    total_sleep_min = deep_min + rem_min + light_min

    time_in_bed_min = total_sleep_min + awake_min
    sleep_efficiency_pct = (total_sleep_min / time_in_bed_min * 100) if time_in_bed_min else None

    deep_pct = (deep_min / total_sleep_min * 100) if total_sleep_min else None
    rem_pct  = (rem_min  / total_sleep_min * 100) if total_sleep_min else None

    sleep_date = None
    if sleep_end_dt:
        sleep_date = sleep_end_dt.strftime("%Y-%m-%d")
    elif payload.get("timestamp"):
        ts = _parse_dt(payload["timestamp"])
        if ts:
            sleep_date = ts.strftime("%Y-%m-%d")

    # ── Heart rate ──────────────────────────────────────────────────────
    overnight_bpms, daytime_bpms, all_bpms = [], [], []
    for e in (payload.get("heart_rate") or []):
        if not isinstance(e, dict):
            continue
        bpm = e.get("bpm")
        if bpm is None:
            continue
        bpm = int(bpm)
        all_bpms.append(bpm)
        t = _parse_dt(e.get("time"))
        if t is None:
            continue
        h = t.hour
        if h >= 21 or h < 8:
            overnight_bpms.append(bpm)
        else:
            daytime_bpms.append(bpm)

    resting_hr = None
    if overnight_bpms:
        s = sorted(overnight_bpms)
        idx = max(0, int(len(s) * 0.10))
        resting_hr = round(sum(s[:idx + 1]) / (idx + 1), 1)
    elif all_bpms:
        resting_hr = float(min(all_bpms))

    avg_hr_overnight = round(sum(overnight_bpms) / len(overnight_bpms), 1) if overnight_bpms else None
    avg_hr_day       = round(sum(daytime_bpms)   / len(daytime_bpms),   1) if daytime_bpms   else None

    if all_bpms:
        n = len(all_bpms)
        mean = sum(all_bpms) / n
        var = sum((b - mean) ** 2 for b in all_bpms) / n
        hr_std = round(var ** 0.5, 2)
        hr_min = min(all_bpms)
        hr_max = max(all_bpms)
    else:
        hr_std = hr_min = hr_max = None

    # ── SpO2 ────────────────────────────────────────────────────────────
    spo2_vals = [float(e["percentage"]) for e in (payload.get("oxygen_saturation") or [])
                 if isinstance(e, dict) and "percentage" in e]
    spo2_avg = round(sum(spo2_vals) / len(spo2_vals), 1) if spo2_vals else None
    spo2_min = round(min(spo2_vals), 1) if spo2_vals else None

    # ── Steps ───────────────────────────────────────────────────────────
    steps = sum(int(e.get("count") or 0) for e in (payload.get("steps") or [])
                if isinstance(e, dict))
    if steps == 0:
        steps = None

    out = {
        "entry_id":             entry_id,
        "person_id":            person_id,
        "sleep_date":           sleep_date,
        "total_sleep_min":      round(total_sleep_min, 2) if total_sleep_min else None,
        "deep_min":             round(deep_min, 2)  if deep_min  else None,
        "rem_min":              round(rem_min, 2)   if rem_min   else None,
        "light_min":            round(light_min, 2) if light_min else None,
        "awake_min":            round(awake_min, 2) if awake_min else None,
        "deep_pct":             round(deep_pct, 2)  if deep_pct  is not None else None,
        "rem_pct":              round(rem_pct, 2)   if rem_pct   is not None else None,
        "sleep_efficiency_pct": round(sleep_efficiency_pct, 2) if sleep_efficiency_pct else None,
        "resting_hr":           resting_hr,
        "avg_hr_overnight":     avg_hr_overnight,
        "avg_hr_day":           avg_hr_day,
        "hr_std":               hr_std,
        "hr_min":               hr_min,
        "hr_max":               hr_max,
        "spo2_avg":             spo2_avg,
        "spo2_min":             spo2_min,
        "steps":                steps,
    }

    if person_meta:
        out["age"]       = person_meta.get("age")
        out["gender"]    = person_meta.get("sex")
        out["height_cm"] = person_meta.get("height_cm")
        out["weight_kg"] = person_meta.get("weight_kg")

    return out


FEATURE_COLUMNS = [
    "entry_id", "person_id", "age", "gender", "height_cm", "weight_kg",
    "sleep_date",
    "total_sleep_min", "deep_min", "rem_min", "light_min", "awake_min",
    "deep_pct", "rem_pct", "sleep_efficiency_pct",
    "resting_hr", "avg_hr_overnight", "avg_hr_day", "hr_std", "hr_min", "hr_max",
    "spo2_avg", "spo2_min", "steps",
    "energy_score",
]
