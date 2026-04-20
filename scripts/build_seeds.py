"""
Build the 4 seed payloads in dataset/seed/.

- surya.json: a copy of one of Surya's real watch payloads (the canonical
  "real" example). We pick the first full-fidelity one we find.
- yashu.json / dad.json / mom.json: synthesized with apply_variance=False so
  they represent each person's baseline night (no jitter).

Run:
    venv/Scripts/python scripts/build_seeds.py
"""
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthesize import synthesize_payload  # noqa: E402

ROOT         = os.path.dirname(HERE)
WEBHOOK_LOGS = os.path.join(ROOT, "webhook_logs")
DATASET_DIR  = os.path.join(ROOT, "dataset")
SEED_DIR     = os.path.join(DATASET_DIR, "seed")
PEOPLE_FILE  = os.path.join(DATASET_DIR, "people.json")


def _find_surya_real_payload() -> str | None:
    """Find the first full-fidelity real payload in webhook_logs."""
    candidates = []
    for root, _dirs, files in os.walk(WEBHOOK_LOGS):
        if "dataset" in root:
            continue
        for f in files:
            if not f.endswith(".json"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and data.get("sleep") and data.get("heart_rate"):
                candidates.append((os.path.getsize(path), path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    os.makedirs(SEED_DIR, exist_ok=True)

    with open(PEOPLE_FILE, "r", encoding="utf-8") as f:
        people = json.load(f)

    # Surya: copy a real payload
    surya_dst = os.path.join(SEED_DIR, "surya.json")
    surya_src = _find_surya_real_payload()
    if surya_src:
        shutil.copy2(surya_src, surya_dst)
        print(f"surya.json: copied from real payload {os.path.relpath(surya_src, ROOT)}")
    else:
        rng = random.Random(42)
        payload = synthesize_payload(people["surya"],
                                     datetime(2026, 4, 1, tzinfo=timezone.utc),
                                     rng, apply_variance=False)
        with open(surya_dst, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("surya.json: no real payload found — synthesized from baseline")

    # Yashu / Dad / Mom: synthesize baseline (no jitter)
    for person_id in ["yashu", "dad", "mom"]:
        rng = random.Random({"yashu": 100, "dad": 200, "mom": 300}[person_id])
        payload = synthesize_payload(people[person_id],
                                     datetime(2026, 4, 15, tzinfo=timezone.utc),
                                     rng, apply_variance=False)
        dst = os.path.join(SEED_DIR, f"{person_id}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        n_stages = len(payload["sleep"][0]["stages"])
        n_hr     = len(payload["heart_rate"])
        n_spo2   = len(payload["oxygen_saturation"])
        print(f"{person_id}.json: synthesized "
              f"({n_stages} stages, {n_hr} HR readings, {n_spo2} SpO2 readings)")


if __name__ == "__main__":
    main()
