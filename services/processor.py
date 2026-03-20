"""
Payload processor: maps raw watch webhook JSON -> WatchReading fields.

Confirmed payload shape (from actual watch data):
{
  "timestamp": "2026-03-17T07:22:30.971814Z",
  "app_version": "1.0",
  "steps": [
    {"count": 3815, "start_time": "...", "end_time": "..."},
    {"count": 344,  "start_time": "...", "end_time": "..."}
  ],
  "sleep": [
    {
      "session_end_time": "2026-03-16T15:59:00Z",
      "duration_seconds": 6000,
      "stages": [
        {"stage": "4", "start_time": "...", "end_time": "...", "duration_seconds": 900},
        ...
      ]
    },
    {
      "session_end_time": "2026-03-17T02:43:30Z",
      "duration_seconds": 19890,
      "stages": [...]
    }
  ],
  "heart_rate": [{"bpm": 80, "time": "2026-03-16T16:35:00Z"}, ...],
  "oxygen_saturation": [{"percentage": 95, "time": "..."}, ...]
}

Stage codes (come as strings in this payload):
  "1" = Awake
  "2" = Light sleep
  "3" = REM
  "4" = Deep sleep
  "5" = REM (alternate code used by some Wear OS watches)
  "6" = Light sleep (alternate code)
"""
import json
from datetime import datetime, timezone

STAGE_DEEP  = {"5"}
STAGE_REM   = {"6"}
STAGE_LIGHT = {"4"}
STAGE_AWAKE = {"1"}


def parse_datetime(value):
    if not value:
        return None
    value = str(value).replace("Z", "+00:00")
    # Handle microseconds
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    value = value.replace("+00:00", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def process_payload(raw: dict, employee_id: str) -> dict:
    fields = _extract_fields(raw)
    fields["employee_id"] = employee_id
    fields["raw_payload"] = json.dumps(raw)
    return fields


def _extract_fields(raw: dict) -> dict:

    # ── Timestamp / sleep date ───────────────────────────────
    ts = parse_datetime(raw.get("timestamp"))
    # Use date of the main sleep session end, fallback to timestamp
    sleep_date = None

    # ── Sleep: iterate ALL sessions, sum up stage durations ──
    deep_sec = rem_sec = light_sec = awake_sec = 0
    sleep_start_dt = None
    sleep_end_dt = None

    sleep_sessions = raw.get("sleep") or []
    for session in sleep_sessions:
        # Track overall sleep window
        session_end = parse_datetime(session.get("session_end_time"))
        if session_end:
            if sleep_end_dt is None or session_end > sleep_end_dt:
                sleep_end_dt = session_end

        for stage in (session.get("stages") or []):
            code = str(stage.get("stage", "")).strip()
            dur  = int(stage.get("duration_seconds") or 0)

            if code in STAGE_DEEP:
                deep_sec  += dur
            elif code in STAGE_REM:
                rem_sec   += dur
            elif code in STAGE_LIGHT:
                light_sec += dur
            elif code in STAGE_AWAKE:
                awake_sec += dur

            # Track earliest stage start as sleep_start
            stage_start = parse_datetime(stage.get("start_time"))
            if stage_start:
                if sleep_start_dt is None or stage_start < sleep_start_dt:
                    sleep_start_dt = stage_start

    deep_min  = deep_sec  / 60
    rem_min   = rem_sec   / 60
    light_min = light_sec / 60
    awake_min = awake_sec / 60
    total_sleep_min = deep_min + rem_min + light_min  # excludes awake time

    # Sleep date = date when the main sleep session ended (morning)
    if sleep_end_dt:
        sleep_date = sleep_end_dt.strftime("%Y-%m-%d")
    elif ts:
        sleep_date = ts.strftime("%Y-%m-%d")
    else:
        sleep_date = datetime.utcnow().strftime("%Y-%m-%d")

    # ── Steps: sum ALL entries ────────────────────────────────
    steps_entries = raw.get("steps") or []
    total_steps = sum(int(e.get("count") or 0) for e in steps_entries if isinstance(e, dict))

    # ── Heart rate ───────────────────────────────────────────
    # Filter to overnight readings only (21:00 - 08:00) for true resting HR
    # Full dataset includes daytime active readings which skew the average
    hr_entries = raw.get("heart_rate") or []
    overnight_bpms = []
    all_bpms = []

    for e in hr_entries:
        if not isinstance(e, dict):
            continue
        bpm = e.get("bpm")
        if bpm is None:
            continue
        bpm = int(bpm)
        all_bpms.append(bpm)

        t = parse_datetime(e.get("time"))
        if t:
            hour = t.hour
            # Overnight = 21:00 to 08:00
            if hour >= 21 or hour < 8:
                overnight_bpms.append(bpm)

    # Resting HR = 10th percentile of overnight readings (filters out movement spikes)
    resting_hr = None
    if overnight_bpms:
        sorted_bpms = sorted(overnight_bpms)
        p10_idx = max(0, int(len(sorted_bpms) * 0.10))
        resting_hr = round(sum(sorted_bpms[:max(1, p10_idx+1)]) / max(1, p10_idx+1), 1)
    elif all_bpms:
        resting_hr = min(all_bpms)  # fallback

    avg_hr = round(sum(overnight_bpms) / len(overnight_bpms), 1) if overnight_bpms else (
             round(sum(all_bpms) / len(all_bpms), 1) if all_bpms else None)

    # ── SpO2: average all readings ────────────────────────────
    spo2_entries = raw.get("oxygen_saturation") or []
    spo2_vals = [float(e["percentage"]) for e in spo2_entries
                 if isinstance(e, dict) and "percentage" in e]
    spo2 = round(sum(spo2_vals) / len(spo2_vals), 1) if spo2_vals else None

    return {
        "sleep_date":           sleep_date,
        "sleep_start":          sleep_start_dt,
        "sleep_end":            sleep_end_dt,
        "total_sleep_minutes":  round(total_sleep_min, 2) if total_sleep_min else None,
        "deep_sleep_minutes":   round(deep_min, 2)  if deep_min  else None,
        "rem_sleep_minutes":    round(rem_min, 2)   if rem_min   else None,
        "light_sleep_minutes":  round(light_min, 2) if light_min else None,
        "awake_minutes":        round(awake_min, 2) if awake_min else None,
        "sleep_score":          None,
        "resting_heart_rate":   _to_float(resting_hr),
        "avg_heart_rate":       _to_float(avg_hr),
        "hrv":                  None,
        "spo2":                 _to_float(spo2),
        "steps":                total_steps if total_steps > 0 else None,
        "active_minutes":       None,
        "calories_burned":      None,
        "stress_score":         None,
    }


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def split_payload_by_sleep_sessions(raw: dict) -> list:
    """
    Split a payload containing multiple sleep sessions into individual payloads.
    Each returned payload contains one sleep session plus all other sensor data.
    If payload has 0 or 1 sleep sessions, returns [raw] unchanged.
    
    Returns:
        List of payload dicts, one per sleep session (in chronological order)
    """
    sleep_sessions = raw.get("sleep") or []
    
    # If 0 or 1 sessions, return as-is
    if len(sleep_sessions) <= 1:
        return [raw]
    
    # Split into individual payloads
    payloads = []
    for session in sleep_sessions:
        payload = raw.copy()
        payload["sleep"] = [session]  # Only this session
        payloads.append(payload)
    
    return payloads


def extract_heart_rate_readings(raw: dict) -> list:
    """
    Extract individual heart rate readings from payload.
    Returns list of dicts with: timestamp (datetime), bpm (int)
    """
    readings = []
    for entry in (raw.get("heart_rate") or []):
        if not isinstance(entry, dict):
            continue
        
        timestamp = parse_datetime(entry.get("time"))
        bpm = entry.get("bpm")
        
        if timestamp and bpm is not None:
            readings.append({
                "timestamp": timestamp,
                "bpm": int(bpm)
            })
    
    return readings


def extract_oxygen_readings(raw: dict) -> list:
    """
    Extract individual SpO2 readings from payload.
    Returns list of dicts with: timestamp (datetime), percentage (float)
    """
    readings = []
    for entry in (raw.get("oxygen_saturation") or []):
        if not isinstance(entry, dict):
            continue
        
        timestamp = parse_datetime(entry.get("time"))
        percentage = entry.get("percentage")
        
        if timestamp and percentage is not None:
            readings.append({
                "timestamp": timestamp,
                "percentage": float(percentage)
            })
    
    return readings


def extract_step_readings(raw: dict) -> list:
    """
    Extract individual step readings from payload.
    Returns list of dicts with: start_time (datetime), end_time (datetime), count (int)
    """
    readings = []
    for entry in (raw.get("steps") or []):
        if not isinstance(entry, dict):
            continue
        
        start_time = parse_datetime(entry.get("start_time"))
        end_time = parse_datetime(entry.get("end_time"))
        count = entry.get("count")
        
        if start_time and end_time and count is not None:
            readings.append({
                "start_time": start_time,
                "end_time": end_time,
                "count": int(count)
            })
    
    return readings