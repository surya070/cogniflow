# Detailed Readings Storage - Complete Implementation

## What Was Fixed

**Before:** BPM, SpO2, steps data were being **discarded** after aggregation
- 705 heart rate readings → stored only 2 values (resting HR, avg HR)
- Individual SpO2 readings → stored only 1 average value
- Step readings → stored only total sum

**After:** ALL detailed readings are now **stored and indexed**
- ✓ 8,265 heart rate readings (with timestamps)
- ✓ 13 SpO2 readings (with timestamps)
- ✓ 13 step readings (with time windows)

## New Database Tables

### HeartRateReading (`heart_rate_readings`)
```
id (PRIMARY KEY)
├─ reading_id (indexed, links to WatchReading)
├─ employee_id (indexed)
├─ timestamp (indexed) ← Each reading has a timestamp
└─ bpm (integer) ← Beats per minute value
```
**Use:** Track heart rate variations throughout the day/night, calculate percentiles, detect anomalies

### OxygenSaturationReading (`oxygen_saturation_readings`)
```
id (PRIMARY KEY)
├─ reading_id (indexed, links to WatchReading)
├─ employee_id (indexed)
├─ timestamp (indexed) ← Each reading has a timestamp
└─ percentage (float) ← SpO2 percentage value
```
**Use:** Monitor blood oxygen levels over time, detect dips, trend analysis

### StepReading (`step_readings`)
```
id (PRIMARY KEY)
├─ reading_id (indexed, links to WatchReading)
├─ employee_id (indexed)
├─ start_time (indexed) ← Time window start
├─ end_time (datetime) ← Time window end
└─ count (integer) ← Steps in that period
```
**Use:** Track activity patterns, hourly/daily step counts, activity timing

## Data Relationships

```
WatchReading (one sleep session)
├─ → HeartRateReading (705 entries per session)
├─ → OxygenSaturationReading (2-3 entries per session)
├─ → StepReading (2-3 entries per session)
└─ → DailyAnalysis (LLM analysis)
```

## What Gets Stored Now

### Webhook Processing
When a webhook arrives:
1. **WatchReading** table: Aggregated metrics
   - resting_heart_rate (10th percentile)
   - avg_heart_rate (overnight average)
   - spo2 (average)
   - steps (total)

2. **HeartRateReading** table: Individual readings
   - All 705+ readings with timestamps
   - Indexed by reading_id and employee_id
   - Can query: all readings for a session, trend analysis, percentile calculations

3. **OxygenSaturationReading** table: All SpO2 readings
   - Each reading with exact timestamp
   - Indexed for fast queries

4. **StepReading** table: Step count windows
   - Each count window with start/end times
   - Can calculate hourly/daily step patterns

## Existing Data

✓ Historical data populated:
- 5 WatchReading records
- 8,265 HeartRateReading records
- 13 OxygenSaturationReading records
- 13 StepReading records

## Code Changes

### 1. models.py
Added three new model classes:
- `HeartRateReading`
- `OxygenSaturationReading`
- `StepReading`

### 2. processor.py
Added extraction functions:
- `extract_heart_rate_readings()` - parses all HR readings
- `extract_oxygen_readings()` - parses all SpO2 readings
- `extract_step_readings()` - parses all step readings

### 3. app.py
Modified `_run_pipeline()` to:
1. Create the main WatchReading (aggregated)
2. Extract detailed readings from raw payload
3. Delete old detailed readings (if updating)
4. Store ALL new detailed readings in separate tables

## Future Use Cases

Now that detailed readings are stored, you can:

✓ **Analytics**: Plot heart rate throughout sleep, detect trends
✓ **Alerts**: Set thresholds on individual readings (e.g., SpO2 < 90%)
✓ **Detailed History**: See minute-by-minute data, not just averages
✓ **Anomaly Detection**: Identify unusual patterns in heart rate variations
✓ **Activity Analysis**: Hourly step counts, activity windows
✓ **Predictive Models**: Train ML models on timestamped readings

## Example Query

```python
# All heart rates for a session during sleep
readings = HeartRateReading.query.filter_by(reading_id=123).all()

# Heart rates above threshold
high_hr = HeartRateReading.query.filter(
    HeartRateReading.reading_id == 123,
    HeartRateReading.bpm > 100
).all()

# SpO2 measurements for an employee
spo2_data = OxygenSaturationReading.query.filter_by(
    employee_id="emp_001"
).order_by(OxygenSaturationReading.timestamp).all()
```
