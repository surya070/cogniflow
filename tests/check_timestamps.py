import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from app import app, db
from models import HeartRateReading, WatchReading

with app.app_context():
    # Check all readings
    readings = WatchReading.query.all()
    print("All WatchReadings in database:")
    for r in readings:
        hr_count = HeartRateReading.query.filter_by(reading_id=r.id).count()
        print(f"  ID {r.id}: {r.employee_id} | Date: {r.sleep_date} | HR readings: {hr_count}")
    
    print("\n\nHeart rate readings for each reading:")
    for r in readings:
        hrs = HeartRateReading.query.filter_by(reading_id=r.id).order_by(HeartRateReading.timestamp).all()
        if hrs:
            first = hrs[0]
            last = hrs[-1]
            print(f"\nReading {r.id} ({r.sleep_date}):")
            print(f"  First HR: {first.timestamp} ({first.bpm} bpm)")
            print(f"  Last HR:  {last.timestamp} ({last.bpm} bpm)")
            print(f"  Total: {len(hrs)} readings")
            
            # Check date distribution
            date_counts = {}
            for hr in hrs:
                date_key = hr.timestamp.date()
                date_counts[date_key] = date_counts.get(date_key, 0) + 1
            
            print(f"  Distribution by date:")
            for date, count in sorted(date_counts.items()):
                print(f"    {date}: {count} readings")
