import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app
from models import WatchReading, HeartRateReading

with app.app_context():
    reading = WatchReading.query.filter_by(sleep_date='2026-03-18').first()
    if reading:
        print(f"Reading {reading.id} (2026-03-18):")
        print(f"  Sleep start: {reading.sleep_start}")
        print(f"  Sleep end: {reading.sleep_end}")
        
        hrs = HeartRateReading.query.filter_by(reading_id=reading.id).filter(
            HeartRateReading.timestamp >= reading.sleep_start,
            HeartRateReading.timestamp <= reading.sleep_end
        ).all()
        print(f"  HR readings in window: {len(hrs)}")
        
        if hrs:
            print(f"  Time span: {hrs[0].timestamp} to {hrs[-1].timestamp}")
            
            from collections import defaultdict
            hours = defaultdict(int)
            for hr in hrs:
                hour = hr.timestamp.strftime("%H:00")
                hours[hour] += 1
            
            print(f"  Hours with data: {len(hours)}")
            for hour in sorted(hours.keys()):
                print(f"    {hour}: {hours[hour]} readings")
