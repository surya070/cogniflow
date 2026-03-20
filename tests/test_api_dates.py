import requests
import json

# Test the HR API for March 18 and 19
for date in ['2026-03-18', '2026-03-19']:
    resp = requests.get(f'http://localhost:5000/api/employee/unknown/heart-rate?date={date}')
    data = resp.json()
    
    print(f"\n{'='*50}")
    print(f"Date: {date}")
    print(f"{'='*50}")
    print(f"Reading ID: {data.get('reading_id')}")
    print(f"Sleep window: {data.get('sleep_start')} to {data.get('sleep_end')}")
    print(f"Total HR readings: {data.get('total_readings')}")
    print(f"Hours with data: {len(data.get('hourly', []))}")
    print(f"\nHourly breakdown:")
    for hour in data.get('hourly', []):
        print(f"  {hour['hour']}: {hour['avg']:.1f} avg ({hour['count']} readings)")
