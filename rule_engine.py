"""
Rule-based cognitive state engine.
Based on established sleep science thresholds.
Returns cognitive_score, fatigue_level, recovery_state.
"""

def score_cognitive_state(reading):
    """
    Input:  a WatchReading ORM object (attributes already populated)
    Output: dict with cognitive_score, fatigue_level, recovery_state, score_details
    """
    points = 0
    max_points = 0
    details = []

    # ── 1. Total Sleep Duration ──────────────────────────────────────────────
    # Optimal: 7–9 hrs (420–540 min)
    sleep_hrs = (reading.total_sleep_minutes or 0) / 60
    max_points += 30
    if sleep_hrs >= 7:
        points += 30
        details.append(f"Sleep duration {sleep_hrs:.1f}h ✓ (optimal)")
    elif sleep_hrs >= 6:
        points += 18
        details.append(f"Sleep duration {sleep_hrs:.1f}h ~ (borderline)")
    elif sleep_hrs >= 5:
        points += 8
        details.append(f"Sleep duration {sleep_hrs:.1f}h ✗ (insufficient)")
    else:
        points += 0
        details.append(f"Sleep duration {sleep_hrs:.1f}h ✗✗ (severely insufficient)")

    # ── 2. Deep Sleep % ──────────────────────────────────────────────────────
    # Optimal: ≥ 20% of total sleep
    max_points += 20
    if reading.total_sleep_minutes and reading.deep_sleep_minutes:
        deep_pct = (reading.deep_sleep_minutes / reading.total_sleep_minutes) * 100
        if deep_pct >= 20:
            points += 20
            details.append(f"Deep sleep {deep_pct:.0f}% ✓")
        elif deep_pct >= 13:
            points += 10
            details.append(f"Deep sleep {deep_pct:.0f}% ~ (below optimal)")
        else:
            points += 0
            details.append(f"Deep sleep {deep_pct:.0f}% ✗ (poor)")
    else:
        points += 10  # neutral if data missing
        details.append("Deep sleep data unavailable")

    # ── 3. REM Sleep % ───────────────────────────────────────────────────────
    # Optimal: ≥ 20% of total sleep
    max_points += 15
    if reading.total_sleep_minutes and reading.rem_sleep_minutes:
        rem_pct = (reading.rem_sleep_minutes / reading.total_sleep_minutes) * 100
        if rem_pct >= 20:
            points += 15
            details.append(f"REM sleep {rem_pct:.0f}% ✓")
        elif rem_pct >= 15:
            points += 8
            details.append(f"REM sleep {rem_pct:.0f}% ~ (slightly low)")
        else:
            points += 0
            details.append(f"REM sleep {rem_pct:.0f}% ✗")
    else:
        points += 7
        details.append("REM data unavailable")

    # ── 4. Resting Heart Rate ────────────────────────────────────────────────
    # Optimal: 50–70 bpm
    max_points += 15
    rhr = reading.resting_heart_rate
    if rhr:
        if 50 <= rhr <= 70:
            points += 15
            details.append(f"Resting HR {rhr:.0f} bpm ✓")
        elif rhr <= 80:
            points += 8
            details.append(f"Resting HR {rhr:.0f} bpm ~ (slightly elevated)")
        else:
            points += 0
            details.append(f"Resting HR {rhr:.0f} bpm ✗ (elevated — stress/fatigue signal)")
    else:
        points += 7
        details.append("Resting HR unavailable")

    # ── 5. HRV ───────────────────────────────────────────────────────────────
    # Higher HRV = better recovery. >50ms is good, <30ms is concerning
    max_points += 10
    hrv = reading.hrv
    if hrv:
        if hrv >= 50:
            points += 10
            details.append(f"HRV {hrv:.0f}ms ✓ (good recovery)")
        elif hrv >= 30:
            points += 5
            details.append(f"HRV {hrv:.0f}ms ~ (moderate recovery)")
        else:
            points += 0
            details.append(f"HRV {hrv:.0f}ms ✗ (low — poor recovery)")
    else:
        points += 5
        details.append("HRV unavailable")

    # ── 6. SpO2 ──────────────────────────────────────────────────────────────
    # Should be ≥ 95%
    max_points += 10
    spo2 = reading.spo2
    if spo2:
        if spo2 >= 95:
            points += 10
            details.append(f"SpO2 {spo2:.0f}% ✓")
        elif spo2 >= 90:
            points += 4
            details.append(f"SpO2 {spo2:.0f}% ~ (slightly low)")
        else:
            points += 0
            details.append(f"SpO2 {spo2:.0f}% ✗ (low oxygen)")
    else:
        points += 5
        details.append("SpO2 unavailable")

    # ── Compute final score ──────────────────────────────────────────────────
    pct = (points / max_points) * 100

    if pct >= 75:
        cognitive_score = "high"
        fatigue_level   = "low"
        recovery_state  = "recovered"
    elif pct >= 50:
        cognitive_score = "moderate"
        fatigue_level   = "moderate"
        recovery_state  = "partial"
    else:
        cognitive_score = "low"
        fatigue_level   = "high"
        recovery_state  = "poor"

    return {
        "cognitive_score": cognitive_score,
        "fatigue_level":   fatigue_level,
        "recovery_state":  recovery_state,
        "pct":             round(pct, 1),
        "details":         details,
    }
