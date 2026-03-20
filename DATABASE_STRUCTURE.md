# Cogniflow Database Structure & Deduplication

## Data Storage

### WatchReading Table (`watch_readings`)
One row per sleep session with the following key fields:

```
id (PRIMARY KEY)
├─ employee_id (indexed)
├─ received_at (timestamp when data arrived)
├─ sleep_date (YYYY-MM-DD format)
├─ sleep_start (UTC datetime)
├─ sleep_end (UTC datetime) ← NEW: Used for deduplication
├─ total_sleep_minutes
├─ deep_sleep_minutes
├─ rem_sleep_minutes
├─ light_sleep_minutes
├─ awake_minutes
├─ resting_heart_rate
├─ avg_heart_rate
├─ hrv
├─ spo2
├─ steps
├─ raw_payload (original JSON stored for debugging)
├─ cognitive_score
├─ fatigue_level
├─ recovery_state
```

### DailyAnalysis Table (`daily_analyses`)
LLM-generated analysis linked to each WatchReading:

```
id (PRIMARY KEY)
├─ reading_id (FOREIGN KEY → WatchReading.id)
├─ employee_id
├─ generated_at
├─ summary (LLM text)
├─ insights (JSON)
├─ task_allocation (JSON)
├─ warnings (JSON)
└─ full_response (raw LLM output)
```

## Deduplication Strategy

### How It Works

When a new webhook arrives:

1. **Payload is split** into individual sleep sessions (if multiple exist)
2. **For each session**, system searches for existing record using:
   ```
   (employee_id, sleep_end) ← UNIQUE KEY
   ```
3. **If found**: Updates existing record with new data
4. **If not found**: Creates new record

### Example: March 18 with 2 Sessions

**Webhook received with 2 sleep sessions:**
- Session A: 21:05 to 03:55 (night sleep)
- Session B: 13:14 to 15:51 (afternoon nap)

**Processing:**
1. Payload split into 2 separate payloads (one per session)
2. Session A: Search by `(emp_001, 2026-03-18 03:55:00)` → finds ID 5, updates it
3. Session B: Search by `(emp_001, 2026-03-18 15:51:00)` → finds ID 6, updates it

**Result:** 2 distinct records, no duplicates, no cross-contamination

### Overlapping Data Handling

**Scenario:** Same webhook received twice with same sleep sessions

**Behavior:**
- First time: Creates new records (if not exist)
- Second time: 
  - Searches by `sleep_end` timestamp
  - Finds exact same session
  - Updates existing record instead of creating duplicate
  - **No new records created** ✓

**Example:**
```
First webhook with March 18 session (ends 03:55):
  → Creates ID 5

Second webhook with same March 18 session:
  → Searches for (emp_001, 2026-03-18 03:55:00)
  → Finds ID 5
  → Updates ID 5 (overwrites old data with new metrics)
  → No duplicate created ✓
```

## Fallback Logic

If `sleep_end` is missing:
1. Falls back to `sleep_date`
2. Only uses fallback if exactly 1 record exists for that date
3. Prevents accidental overwrites when multiple sessions per day

## Recent Fix

**Before:** Deduplication used only `(employee_id, sleep_date)`
- Problem: Would overwrite one session when multiple existed on same day
- March 18 morning session could be replaced by afternoon session

**After:** Deduplication uses `(employee_id, sleep_end)`
- Solution: Each session uniquely identified by its end timestamp
- March 18 morning and afternoon sessions stay separate ✓

## Multiple Employee Support

System supports multiple employees simultaneously:
- Each employee's data is isolated
- Deduplication checks: `(employee_id, sleep_end)`
- No cross-employee data mixing
