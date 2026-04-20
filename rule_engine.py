"""
Rule-based cognitive state and stress scoring engine.
"""


def score_cognitive_state(reading):
    """
    Input:  a WatchReading ORM object
    Output: dict with cognitive_score, fatigue_level, recovery_state, pct, details
    pct is 0–100 and used as a readiness fallback when no ML formula is available.
    """
    points = 0
    max_points = 0
    details = []

    # 1. Total Sleep Duration — optimal: 7–9 hrs
    sleep_hrs = (reading.total_sleep_minutes or 0) / 60
    max_points += 30
    if sleep_hrs >= 7:
        points += 30
        details.append(f"Sleep {sleep_hrs:.1f}h — optimal")
    elif sleep_hrs >= 6:
        points += 18
        details.append(f"Sleep {sleep_hrs:.1f}h — borderline")
    elif sleep_hrs >= 5:
        points += 8
        details.append(f"Sleep {sleep_hrs:.1f}h — insufficient")
    else:
        details.append(f"Sleep {sleep_hrs:.1f}h — severely insufficient")

    # 2. Deep Sleep % — optimal: >=20%
    max_points += 20
    if reading.total_sleep_minutes and reading.deep_sleep_minutes:
        deep_pct = (reading.deep_sleep_minutes / reading.total_sleep_minutes) * 100
        if deep_pct >= 20:
            points += 20
            details.append(f"Deep sleep {deep_pct:.0f}% — good")
        elif deep_pct >= 13:
            points += 10
            details.append(f"Deep sleep {deep_pct:.0f}% — below optimal")
        else:
            details.append(f"Deep sleep {deep_pct:.0f}% — poor")
    else:
        points += 10
        details.append("Deep sleep data unavailable")

    # 3. REM Sleep % — optimal: >=20%
    max_points += 15
    if reading.total_sleep_minutes and reading.rem_sleep_minutes:
        rem_pct = (reading.rem_sleep_minutes / reading.total_sleep_minutes) * 100
        if rem_pct >= 20:
            points += 15
            details.append(f"REM sleep {rem_pct:.0f}% — good")
        elif rem_pct >= 15:
            points += 8
            details.append(f"REM sleep {rem_pct:.0f}% — slightly low")
        else:
            details.append(f"REM sleep {rem_pct:.0f}% — poor")
    else:
        points += 7
        details.append("REM data unavailable")

    # 4. Resting Heart Rate — optimal: 50–70 bpm
    max_points += 15
    rhr = reading.resting_heart_rate
    if rhr:
        if 50 <= rhr <= 70:
            points += 15
            details.append(f"Resting HR {rhr:.0f} bpm — good")
        elif rhr <= 80:
            points += 8
            details.append(f"Resting HR {rhr:.0f} bpm — slightly elevated")
        else:
            details.append(f"Resting HR {rhr:.0f} bpm — elevated")
    else:
        points += 7
        details.append("Resting HR unavailable")

    # 5. HRV — >50ms good, <30ms concerning
    max_points += 10
    hrv = reading.hrv
    if hrv:
        if hrv >= 50:
            points += 10
            details.append(f"HRV {hrv:.0f}ms — good recovery")
        elif hrv >= 30:
            points += 5
            details.append(f"HRV {hrv:.0f}ms — moderate recovery")
        else:
            details.append(f"HRV {hrv:.0f}ms — poor recovery")
    else:
        points += 5
        details.append("HRV unavailable")

    # 6. SpO2 — should be >=95%
    max_points += 10
    spo2 = reading.spo2
    if spo2:
        if spo2 >= 95:
            points += 10
            details.append(f"SpO2 {spo2:.0f}% — normal")
        elif spo2 >= 90:
            points += 4
            details.append(f"SpO2 {spo2:.0f}% — slightly low")
        else:
            details.append(f"SpO2 {spo2:.0f}% — low oxygen")
    else:
        points += 5
        details.append("SpO2 unavailable")

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


def generate_insights(reading) -> list:
    """
    Return a list of actionable insight strings based on the reading's biometrics.
    Used in place of AI analysis.
    """
    insights = []
    score = reading.energy_score or 0
    sleep_hrs = (reading.total_sleep_minutes or 0) / 60

    # Energy tier headline
    if score >= 70:
        insights.append("You're well recovered today — good window for deep work, learning, or high-effort tasks.")
    elif score >= 40:
        insights.append("Moderate recovery. Prioritise focused tasks in the morning when alertness peaks, lighter work after lunch.")
    else:
        insights.append("Low energy today. Stick to routine tasks and avoid high-stakes decisions or creative heavy lifting.")

    # Sleep duration
    if sleep_hrs < 6:
        insights.append(f"Only {sleep_hrs:.1f}h of sleep detected — consider a 20-minute nap mid-day to restore alertness.")
    elif sleep_hrs < 7:
        insights.append(f"{sleep_hrs:.1f}h sleep is below the 7–9h optimum. Aim for an earlier bedtime tonight.")
    elif sleep_hrs >= 9:
        insights.append(f"{sleep_hrs:.1f}h sleep — above average. If you still feel tired, check for poor-quality stages below.")

    # Sleep efficiency
    eff = reading.sleep_efficiency_pct
    if eff and eff < 75:
        insights.append(f"Sleep efficiency is {eff:.0f}% — a lot of time spent awake in bed. Avoid screens and heavy meals in the hour before sleep.")
    elif eff and eff >= 90:
        insights.append(f"Sleep efficiency {eff:.0f}% — excellent. Your sleep quality is very high.")

    # Deep sleep
    if reading.total_sleep_minutes and reading.deep_sleep_minutes:
        deep_pct = (reading.deep_sleep_minutes / reading.total_sleep_minutes) * 100
        if deep_pct < 13:
            insights.append(f"Deep sleep was only {deep_pct:.0f}% — physical recovery may be limited. Avoid alcohol and late exercise.")
        elif deep_pct >= 20:
            insights.append(f"Deep sleep at {deep_pct:.0f}% — strong physical restoration overnight.")

    # REM sleep
    if reading.total_sleep_minutes and reading.rem_sleep_minutes:
        rem_pct = (reading.rem_sleep_minutes / reading.total_sleep_minutes) * 100
        if rem_pct < 15:
            insights.append(f"REM sleep low at {rem_pct:.0f}% — cognitive performance and memory consolidation may be affected.")

    # Resting HR
    rhr = reading.resting_heart_rate
    if rhr:
        if rhr > 80:
            insights.append(f"Resting HR of {rhr:.0f} bpm is elevated — a sign of physiological stress or incomplete recovery.")
        elif rhr <= 55:
            insights.append(f"Resting HR of {rhr:.0f} bpm indicates strong cardiovascular recovery.")

    # SpO2
    spo2 = reading.spo2
    if spo2:
        if spo2 < 92:
            insights.append(f"SpO2 at {spo2:.0f}% is below normal. Ensure good ventilation when sleeping.")
        elif spo2 < 95:
            insights.append(f"SpO2 slightly low at {spo2:.0f}%. Monitor — could affect cognitive stamina.")

    # Steps
    if reading.steps and reading.steps < 3000:
        insights.append("Very low step count yesterday. Even a short walk today can boost mood and afternoon alertness.")

    # Today's recommendation
    if score >= 70:
        insights.append("Today's agenda: schedule demanding meetings, creative work, or complex problem-solving in the morning.")
    elif score >= 40:
        insights.append("Today's agenda: handle communication and collaborative tasks; keep solo deep work short and time-boxed.")
    else:
        insights.append("Today's agenda: clear your calendar where possible. Attend only essential meetings. Rest is recovery.")

    return insights


def compute_stress_score(reading) -> float:
    """
    Derive a stress score 0–100 from biometrics.
    0 = calm/well-recovered, 100 = highly stressed/overloaded.
    Primarily driven by resting HR, sleep duration, and HRV when available.
    """
    score = 40  # neutral baseline

    rhr = reading.resting_heart_rate
    if rhr:
        if rhr <= 55:
            score -= 20
        elif rhr <= 65:
            score -= 10
        elif rhr <= 75:
            score += 5
        elif rhr <= 85:
            score += 18
        else:
            score += 30

    hrv = reading.hrv
    if hrv:
        if hrv >= 60:
            score -= 20
        elif hrv >= 40:
            score -= 10
        elif hrv >= 25:
            score += 5
        else:
            score += 20

    sleep_hrs = (reading.total_sleep_minutes or 0) / 60
    if sleep_hrs >= 8:
        score -= 15
    elif sleep_hrs >= 7:
        score -= 7
    elif sleep_hrs >= 6:
        score += 5
    elif sleep_hrs >= 5:
        score += 15
    else:
        score += 28

    spo2 = reading.spo2
    if spo2:
        if spo2 < 92:
            score += 15
        elif spo2 < 95:
            score += 7

    return float(max(0, min(100, score)))
