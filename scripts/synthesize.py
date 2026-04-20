"""
Synthesizer: turns a person's baseline metadata (from people.json) + a random
seed into a full Health Auto Export-style webhook payload — sleep stages with
per-stage timestamps, heart rate per minute, SpO2 readings, and steps.

Used by build_seeds.py (one call per person) and populate_dataset.py (many
calls per person with different random seeds + jitter).

Nothing is thrown away: the output mirrors the real payload shape so a full
per-minute HR trace, per-timestamp SpO2, and full stage sequence are preserved.
"""
import random
from datetime import datetime, timedelta, timezone


# Stage codes we emit (using the CORRECT mapping — see memory/project_data.md)
STAGE_AWAKE = "1"
STAGE_DEEP  = "4"
STAGE_REM   = "5"
STAGE_LIGHT = "6"

# IST offset — the user is in India. Local times like "23:30" in people.json
# are IST; we convert to UTC for the payload (which uses Z timestamps).
IST_OFFSET = timedelta(hours=5, minutes=30)


def _iso_z(dt: datetime) -> str:
    """Format a UTC datetime as ISO 8601 with a 'Z' suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_local_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def _sleep_window_utc(night_date: datetime, person: dict, rng: random.Random,
                      jitter_minutes: float = 0.0) -> tuple[datetime, datetime]:
    """
    Compute sleep_start (UTC) and sleep_end (UTC) for a given night_date (the
    calendar date of the MORNING when the person wakes up).

    jitter_minutes: symmetric gaussian jitter (sigma in minutes) applied
    separately to start and end times. 0 = deterministic.
    """
    sh, sm = _parse_local_hhmm(person["sleep_start_local"])
    eh, em = _parse_local_hhmm(person["sleep_end_local"])

    # sleep_start is on the evening BEFORE night_date if sleep_start_local >= 18:00
    start_local_date = night_date - timedelta(days=1) if sh >= 18 else night_date
    start_local = datetime(start_local_date.year, start_local_date.month, start_local_date.day,
                           sh, sm, tzinfo=timezone.utc)
    end_local   = datetime(night_date.year, night_date.month, night_date.day,
                           eh, em, tzinfo=timezone.utc)

    if jitter_minutes > 0:
        start_local += timedelta(minutes=rng.gauss(0, jitter_minutes))
        end_local   += timedelta(minutes=rng.gauss(0, jitter_minutes))
        if end_local <= start_local + timedelta(hours=3):
            end_local = start_local + timedelta(hours=6)  # safety

    # Treat sh/eh as IST, convert to UTC.
    start_utc = start_local - IST_OFFSET
    end_utc   = end_local   - IST_OFFSET
    return start_utc, end_utc


def _allocate_stages(total_sleep_min: float, deep_pct: float, rem_pct: float,
                     awake_min: float) -> tuple[float, float, float, float]:
    """Compute minutes for each stage."""
    deep_min  = total_sleep_min * deep_pct / 100
    rem_min   = total_sleep_min * rem_pct  / 100
    light_min = total_sleep_min - deep_min - rem_min
    return deep_min, rem_min, max(light_min, 0.0), awake_min


def _build_stage_sequence(sleep_start: datetime,
                          deep_min: float, rem_min: float,
                          light_min: float, awake_min: float,
                          rng: random.Random) -> tuple[list[dict], datetime]:
    """
    Build a realistic stage sequence spanning the night.

    Shape: N cycles of ~90 min each. Early cycles lean deep; later cycles lean REM.
    Within each cycle: Light → Deep → Light → REM → (maybe brief Awake).

    Returns (stages_list, sleep_end_dt).
    """
    time_in_bed_min = deep_min + rem_min + light_min + awake_min
    n_cycles = max(3, min(6, round(time_in_bed_min / 90)))

    deep_weights  = [1.5, 1.2, 1.0, 0.6, 0.3, 0.2][:n_cycles]
    rem_weights   = [0.3, 0.6, 1.0, 1.3, 1.5, 1.5][:n_cycles]
    dw_sum = sum(deep_weights)
    rw_sum = sum(rem_weights)

    stages = []
    t = sleep_start

    for i in range(n_cycles):
        cyc_deep  = deep_min  * deep_weights[i] / dw_sum
        cyc_rem   = rem_min   * rem_weights[i]  / rw_sum
        cyc_light = light_min / n_cycles
        cyc_awake = awake_min / n_cycles

        # Add small per-segment jitter so stages aren't identical across cycles
        def _j(v, frac=0.1):
            return max(0.0, v * (1 + rng.uniform(-frac, frac)))

        segments = [
            (STAGE_LIGHT, _j(cyc_light * 0.30)),
            (STAGE_DEEP,  _j(cyc_deep)),
            (STAGE_LIGHT, _j(cyc_light * 0.45)),
            (STAGE_REM,   _j(cyc_rem)),
            (STAGE_LIGHT, _j(cyc_light * 0.25)),
        ]
        if cyc_awake > 0.5:
            segments.append((STAGE_AWAKE, _j(cyc_awake)))

        for code, mins in segments:
            if mins <= 0:
                continue
            sec = int(round(mins * 60))
            if sec < 30:
                continue
            stages.append({
                "stage":            code,
                "start_time":       _iso_z(t),
                "end_time":         _iso_z(t + timedelta(seconds=sec)),
                "duration_seconds": sec,
            })
            t += timedelta(seconds=sec)

    return stages, t


def _generate_heart_rate(window_start: datetime, window_end: datetime,
                         sleep_start: datetime, sleep_end: datetime,
                         person: dict, rng: random.Random) -> list[dict]:
    """Per-minute HR from window_start to window_end."""
    resting   = person["baseline_resting_hr"]
    daytime   = person["baseline_daytime_hr"]

    readings = []
    t = window_start
    current = daytime  # seed

    while t < window_end:
        if sleep_start <= t <= sleep_end:
            target = resting
            sigma  = 2.5
        else:
            # gentle diurnal bump in the afternoon
            hour_ist = (t + IST_OFFSET).hour
            daytime_mod = daytime + (3 if 14 <= hour_ist <= 18 else 0)
            target = daytime_mod
            sigma  = 7

        # Autoregressive smoothing so neighbouring minutes don't jump wildly
        current = 0.75 * current + 0.25 * target + rng.gauss(0, sigma)
        current = max(45.0, min(160.0, current))

        readings.append({
            "bpm":  int(round(current)),
            "time": _iso_z(t),
        })
        t += timedelta(minutes=1)

    return readings


def _generate_spo2(window_start: datetime, sleep_end: datetime,
                   person: dict, rng: random.Random) -> list[dict]:
    """A handful of SpO2 readings scattered across the window (matches real data shape)."""
    baseline = person["baseline_spo2"]
    # 3–8 readings, mostly overnight
    n = rng.randint(3, 8)
    readings = []
    for _ in range(n):
        # Distribute across the 12h leading up to wake
        offset_min = rng.randint(0, 12 * 60)
        t = sleep_end - timedelta(minutes=offset_min)
        if t < window_start:
            t = window_start + timedelta(minutes=rng.randint(0, 60))
        val = baseline + rng.gauss(0, 1.2)
        val = max(88.0, min(100.0, val))
        readings.append({
            "percentage": round(val, 1),
            "time":       _iso_z(t),
        })
    readings.sort(key=lambda r: r["time"])
    return readings


def _generate_steps(window_start: datetime, window_end: datetime,
                    person: dict, rng: random.Random) -> list[dict]:
    """3 step entries like the real data: prev-day, main-day, partial-trailing."""
    baseline = person["baseline_daily_steps"]

    def _daily():
        # log-normal-ish daily count
        return max(100, int(rng.lognormvariate(0, 0.35) * baseline))

    # Windows are aligned to 18:30 UTC boundaries (matches real payloads which
    # chop days at 00:00 IST = 18:30 UTC).
    base = window_start.replace(hour=18, minute=30, second=0, microsecond=0)
    if base > window_start:
        base -= timedelta(days=1)

    entries = [
        {"count": _daily(), "start_time": _iso_z(base),                          "end_time": _iso_z(base + timedelta(days=1))},
        {"count": _daily(), "start_time": _iso_z(base + timedelta(days=1)),      "end_time": _iso_z(base + timedelta(days=2))},
        {"count": max(0, int(_daily() * rng.uniform(0.1, 0.7))),
         "start_time": _iso_z(base + timedelta(days=2)),
         "end_time":   _iso_z(window_end)},
    ]
    return entries


def synthesize_payload(person: dict, night_date: datetime,
                       rng: random.Random,
                       apply_variance: bool = True) -> dict:
    """
    Produce one full webhook-style payload for the given person + night.

    night_date: calendar date of the wake-up morning (UTC date is fine — we just
                use it as a reference point).
    apply_variance: if False, uses exact baselines (used by build_seeds.py).
                    if True, jitters every field (used by populate_dataset.py).
    """
    # ── Jitter the baselines ────────────────────────────────────────────
    if apply_variance:
        # Sleep duration jitter around the baseline (σ = 45 min)
        base_start, base_end = _sleep_window_utc(night_date, person, rng, jitter_minutes=30)
        base_duration = (base_end - base_start).total_seconds() / 60
        duration_jitter = rng.gauss(0, 45)
        total_time_in_bed = max(240, base_duration + duration_jitter)  # ≥ 4h
        sleep_start = base_start
        sleep_end   = sleep_start + timedelta(minutes=total_time_in_bed)

        # Age-modulated stage percentages: older → slightly less deep
        age_deep_adj = (person["age"] - 30) * -0.1   # 30yo baseline, older loses deep
        deep_pct = max(5.0, rng.gauss(person["baseline_deep_pct"] + age_deep_adj, 2.0))
        rem_pct  = max(8.0, rng.gauss(person["baseline_rem_pct"], 2.5))
        if deep_pct + rem_pct > 65:
            # keep room for light sleep
            scale = 60 / (deep_pct + rem_pct)
            deep_pct *= scale
            rem_pct  *= scale

        # Awake minutes: 2–6% of time in bed
        awake_min = total_time_in_bed * rng.uniform(0.02, 0.06)
        total_sleep_min = total_time_in_bed - awake_min

        # HR jitter: adjust baseline per-entry before HR generation
        person_jittered = dict(person)
        person_jittered["baseline_resting_hr"] = person["baseline_resting_hr"] + rng.gauss(0, 2.0)
        person_jittered["baseline_daytime_hr"] = person["baseline_daytime_hr"] + rng.gauss(0, 4.0)
        person_jittered["baseline_spo2"]       = person["baseline_spo2"]       + rng.gauss(0, 0.6)

    else:
        sleep_start, sleep_end = _sleep_window_utc(night_date, person, rng, jitter_minutes=0)
        total_time_in_bed = (sleep_end - sleep_start).total_seconds() / 60
        deep_pct = person["baseline_deep_pct"]
        rem_pct  = person["baseline_rem_pct"]
        awake_min = total_time_in_bed * 0.04
        total_sleep_min = total_time_in_bed - awake_min
        person_jittered = person

    deep_min, rem_min, light_min, awake_min = _allocate_stages(
        total_sleep_min, deep_pct, rem_pct, awake_min
    )

    stages, session_end = _build_stage_sequence(
        sleep_start, deep_min, rem_min, light_min, awake_min, rng
    )
    session_duration_sec = int((session_end - sleep_start).total_seconds())

    # HR window: from ~3h before sleep to ~10h after wake (matches real data span)
    hr_window_start = sleep_start - timedelta(hours=3)
    hr_window_end   = session_end + timedelta(hours=10)

    heart_rate = _generate_heart_rate(hr_window_start, hr_window_end,
                                      sleep_start, session_end,
                                      person_jittered, rng)
    spo2 = _generate_spo2(hr_window_start, session_end, person_jittered, rng)
    steps = _generate_steps(hr_window_start, hr_window_end, person_jittered, rng)

    payload = {
        "timestamp":   _iso_z(hr_window_end),
        "app_version": "1.0",
        "steps": steps,
        "sleep": [{
            "session_end_time": _iso_z(session_end),
            "duration_seconds": session_duration_sec,
            "stages":           stages,
        }],
        "heart_rate":        heart_rate,
        "oxygen_saturation": spo2,
    }
    return payload
