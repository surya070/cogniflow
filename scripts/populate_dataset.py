"""
Populate the synthetic dataset.

For each person in people.json, generate N full webhook-style payloads
(preserving per-minute HR, per-timestamp SpO2, stage sequences) with variance
applied to EVERY field, keyed on age/gender.

Output:
  dataset/entries/<person_id>_NNN.json   (full-fidelity payloads)
  dataset/features.csv                   (extracted features, one row per entry, energy_score blank)

Run:
    venv/Scripts/python scripts/populate_dataset.py          # default: 100 per person
    venv/Scripts/python scripts/populate_dataset.py 50       # custom count
"""
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthesize import synthesize_payload
from extract_features import extract_features, FEATURE_COLUMNS

ROOT         = os.path.dirname(HERE)
DATASET_DIR  = os.path.join(ROOT, "dataset")
ENTRIES_DIR  = os.path.join(DATASET_DIR, "entries")
PEOPLE_FILE  = os.path.join(DATASET_DIR, "people.json")
FEATURES_CSV = os.path.join(DATASET_DIR, "features.csv")

DEFAULT_N_PER_PERSON = 100


def main():
    n_per_person = DEFAULT_N_PER_PERSON
    if len(sys.argv) > 1:
        try:
            n_per_person = int(sys.argv[1])
        except ValueError:
            print(f"Invalid count '{sys.argv[1]}', using default {DEFAULT_N_PER_PERSON}")

    os.makedirs(ENTRIES_DIR, exist_ok=True)

    with open(PEOPLE_FILE, "r", encoding="utf-8") as f:
        people = json.load(f)

    rows = []
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for person_id, person in people.items():
        seed_offset = abs(hash(person_id)) % 100_000
        rng = random.Random(seed_offset)
        print(f"\n-- {person_id} ({person['name']}, {person['age']}{person['sex']}) "
              f"-- generating {n_per_person} entries --")

        for i in range(1, n_per_person + 1):
            night_date = base_date + timedelta(days=i * 2)  # spread nights out
            payload = synthesize_payload(person, night_date, rng, apply_variance=True)

            entry_id = f"{person_id}_{i:03d}"
            out_path = os.path.join(ENTRIES_DIR, f"{entry_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            feats = extract_features(payload, person_meta=person,
                                     entry_id=entry_id, person_id=person_id)
            feats["energy_score"] = ""  # filled later by label_dataset.py
            rows.append(feats)

            if i % 25 == 0 or i == n_per_person:
                print(f"  {i}/{n_per_person}")

    # Write features.csv
    with open(FEATURES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FEATURE_COLUMNS})

    print(f"\nWrote {len(rows)} JSON entries to {os.path.relpath(ENTRIES_DIR, ROOT)}")
    print(f"Wrote features.csv ({len(rows)} rows) to {os.path.relpath(FEATURES_CSV, ROOT)}")
    print("energy_score column is empty -- run scripts/label_dataset.py next.")


if __name__ == "__main__":
    main()
