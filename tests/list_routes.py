import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app

# List routes
print("Available routes containing 'heart-rate' or 'api':")
for rule in app.url_map.iter_rules():
    if 'heart-rate' in str(rule.rule) or 'api' in str(rule.rule):
        print(f'  {rule.rule} | Methods: {rule.methods}')
