"""
Gemini-powered analysis and task allocation engine.
"""
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.0-flash")


SYSTEM_PROMPT = """You are Cogniflow, an AI workforce intelligence system.
You analyze an employee's physiological data from their smartwatch and provide:
1. A short human-readable health summary
2. Key insights about their cognitive state today
3. Task allocation recommendations based on their readiness
4. Any burnout or fatigue warnings

Always respond in valid JSON with exactly these keys:
{
  "summary": "2-3 sentence narrative about how the employee is doing today",
  "insights": ["insight 1", "insight 2", "insight 3"],
  "task_allocation": {
    "recommended": ["type of tasks they should do today"],
    "avoid": ["type of tasks to avoid"],
    "reasoning": "why this allocation makes sense"
  },
  "warnings": ["warning 1 if any, else empty list"],
  "readiness_label": "one of: Peak Performance / Good to Go / Proceed Carefully / Rest Recommended"
}
"""


def analyze_reading(reading, score_result):
    """
    reading:      WatchReading ORM object
    score_result: dict from rule_engine.score_cognitive_state()
    Returns:      dict parsed from Gemini JSON response
    """
    sleep_hrs = (reading.total_sleep_minutes or 0) / 60
    deep_pct = (
        round((reading.deep_sleep_minutes / reading.total_sleep_minutes) * 100, 1)
        if reading.total_sleep_minutes and reading.deep_sleep_minutes
        else "N/A"
    )
    rem_pct = (
        round((reading.rem_sleep_minutes / reading.total_sleep_minutes) * 100, 1)
        if reading.total_sleep_minutes and reading.rem_sleep_minutes
        else "N/A"
    )

    user_prompt = f"""
Analyze this employee's health data and provide recommendations:

EMPLOYEE: {reading.employee_id}
DATE: {reading.sleep_date}

SLEEP DATA:
- Total sleep: {sleep_hrs:.1f} hours
- Deep sleep: {deep_pct}%
- REM sleep: {rem_pct}%
- Awake time: {reading.awake_minutes or 'N/A'} minutes

HEART DATA:
- Resting HR: {reading.resting_heart_rate or 'N/A'} bpm
- HRV: {reading.hrv or 'N/A'} ms
- SpO2: {reading.spo2 or 'N/A'}%

ACTIVITY:
- Steps: {reading.steps or 'N/A'}
- Active minutes: {reading.active_minutes or 'N/A'}

STRESS SCORE: {reading.stress_score or 'N/A'} / 100

COGNITIVE ASSESSMENT (rule engine):
- Cognitive score: {score_result['cognitive_score'].upper()}
- Fatigue level: {score_result['fatigue_level'].upper()}
- Recovery state: {score_result['recovery_state'].upper()}
- Readiness %: {score_result['pct']}%

Based on this data, provide your analysis in the required JSON format.
"""

    try:
        response = _model.generate_content(
            [{"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + user_prompt]}]
        )
        raw = response.text.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)

    except Exception as e:
        # Fallback if Gemini fails — still return structured data
        return {
            "summary": f"Analysis unavailable: {str(e)}",
            "insights": score_result["details"],
            "task_allocation": {
                "recommended": _fallback_tasks(score_result["cognitive_score"]),
                "avoid": [],
                "reasoning": "Based on rule engine scoring only.",
            },
            "warnings": [],
            "readiness_label": _fallback_label(score_result["cognitive_score"]),
        }


def _fallback_tasks(cognitive_score):
    mapping = {
        "high":     ["Complex problem solving", "Critical decisions", "Deep focus work", "Code reviews"],
        "moderate": ["Meetings", "Emails", "Documentation", "Collaborative work"],
        "low":      ["Admin tasks", "Light reading", "Routine checklists"],
    }
    return mapping.get(cognitive_score, ["Routine tasks"])


def _fallback_label(cognitive_score):
    mapping = {
        "high":     "Peak Performance",
        "moderate": "Good to Go",
        "low":      "Proceed Carefully",
    }
    return mapping.get(cognitive_score, "Good to Go")
