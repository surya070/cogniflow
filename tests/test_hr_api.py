import requests
import json

# Test the HR API with reading ID 3 (from database)
url = 'http://localhost:5000/api/employee/unknown/heart-rate?reading_id=3'
print(f"Testing: {url}")

resp = requests.get(url)
print(f'Status Code: {resp.status_code}')

if resp.status_code == 200:
    data = resp.json()
    print(f'\nHR API Response:')
    print(f'  Reading ID: {data.get("reading_id")}')
    print(f'  Total Readings: {data.get("total_readings")}')
    print(f'  Resting HR: {data.get("resting_hr")} bpm')
    print(f'  Average HR: {data.get("avg_hr")} bpm')
    print(f'  Hourly Data Points: {len(data.get("hourly", []))}')
    if data.get('hourly'):
        print(f'\n  First 3 hours:')
        for hour in data.get('hourly')[:3]:
            print(f'    {hour}')
else:
    print(f'Error: {resp.text}')
