"""Verify detailed readings storage"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, db
from models import WatchReading, HeartRateReading, OxygenSaturationReading, StepReading

with app.app_context():
    print('=== DETAILED READINGS STORAGE ===\n')
    
    reading = WatchReading.query.first()
    if reading:
        print(f'Example: WatchReading {reading.id} ({reading.sleep_date})')
        print()
        
        # Heart rate
        hr_readings = HeartRateReading.query.filter_by(reading_id=reading.id).limit(5).all()
        print('First 5 Heart Rate readings:')
        for hr in hr_readings:
            time_str = hr.timestamp.strftime('%H:%M:%S')
            print(f'  {time_str}: {hr.bpm} bpm')
        hr_total = HeartRateReading.query.filter_by(reading_id=reading.id).count()
        print(f'  ... ({hr_total} total)')
        
        # SpO2
        spo2_readings = OxygenSaturationReading.query.filter_by(reading_id=reading.id).all()
        print('\nOxygen Saturation readings:')
        for spo2 in spo2_readings:
            time_str = spo2.timestamp.strftime('%H:%M:%S')
            print(f'  {time_str}: {spo2.percentage}%')
        
        # Steps
        step_readings = StepReading.query.filter_by(reading_id=reading.id).all()
        print('\nStep readings:')
        for step in step_readings:
            date_str = step.start_time.date()
            print(f'  {date_str}: {step.count} steps')
    
    print('\n=== TOTAL STORED DATA ===')
    hr_count = HeartRateReading.query.count()
    spo2_count = OxygenSaturationReading.query.count()
    step_count = StepReading.query.count()
    
    print(f'Heart Rate readings: {hr_count}')
    print(f'SpO2 readings: {spo2_count}')
    print(f'Step readings: {step_count}')
