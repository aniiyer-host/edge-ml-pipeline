import csv
import math
from collections import defaultdict
from pathlib import Path

SOURCE = Path("dataset.txt")
OUTPUT = Path("gesture-cmd/server/data/gesture_dataset_clean.csv")

RAW_HEADER = [
    "label", "sample_id", "timestamp_ms", "sample",
    "ax", "ay", "az", "gx", "gy", "gz"
]
OUTPUT_HEADER = [
    "label", "sample_id", "sample", "timestamp_ms",
    "ax", "ay", "az", "gx", "gy", "gz"
]
EXPECTED_LABELS = {
    "idle", "wave_left", "wave_right",
    "flick_up", "flick_down", "wrist_rotate"
}

groups = defaultdict(list)
notes = []
headers = 0

with SOURCE.open(newline="", encoding="utf-8-sig") as f:
    for line_no, raw in enumerate(f, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        row = [x.strip() for x in next(csv.reader([raw]))]

        if row == RAW_HEADER:
            headers += 1
            continue

        if len(row) != 10:
            notes.append((line_no, line))
            continue

        label = row[0]
        try:
            sid = int(row[1])
            sample = int(row[2])
            timestamp = int(row[3])
            sensors = [float(v) for v in row[4:10]]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric row {line_no}: {line}") from exc

        if label not in EXPECTED_LABELS:
            raise ValueError(f"Unexpected label on line {line_no}: {label}")
        if not all(math.isfinite(v) for v in sensors):
            raise ValueError(f"Non-finite sensor value on line {line_no}")

        groups[(label, sid)].append(
            [label, sid, sample, timestamp, *sensors]
        )

for (label, sid), rows in groups.items():
    rows.sort(key=lambda r: r[2])
    indices = [r[2] for r in rows]
    if len(rows) != 50 or indices != list(range(50)):
        raise ValueError(
            f"Invalid window {label}/{sid}: "
            f"{len(rows)} rows, indices={indices}"
        )

labels_found = {label for label, _ in groups}
if labels_found != EXPECTED_LABELS:
    raise ValueError(
        f"Label mismatch: missing={EXPECTED_LABELS-labels_found}, "
        f"unexpected={labels_found-EXPECTED_LABELS}"
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(OUTPUT_HEADER)
    for key in sorted(groups):
        writer.writerows(groups[key])

print("Dataset preparation successful.")
print(f"Source preserved: {SOURCE}")
print(f"Clean CSV: {OUTPUT}")
print(f"Valid windows: {len(groups)}")
print(f"Valid samples: {sum(len(rows) for rows in groups.values())}")
print(f"Repeated headers skipped: {headers}")
print(f"Free-text rows skipped: {len(notes)}")
for label in sorted(EXPECTED_LABELS):
    count = sum(1 for name, _ in groups if name == label)
    print(f"  {label}: {count} windows")
print("Corrected CSV header:", ",".join(OUTPUT_HEADER))
