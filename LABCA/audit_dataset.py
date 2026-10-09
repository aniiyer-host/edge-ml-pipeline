import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED = {
    "idle", "wave_left", "wave_right",
    "flick_up", "flick_down", "wrist_rotate"
}
HEADER = [
    "label", "sample_id", "timestamp_ms", "sample",
    "ax", "ay", "az", "gx", "gy", "gz"
]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
windows = defaultdict(list)
labels = Counter()
notes = []
bad_rows = []
headers = 0
duplicates = 0
seen = set()
numeric_rows = 0
range_violations = []
boundary_hits = []

with open("dataset.txt", newline="", encoding="utf-8-sig") as f:
    for line_no, raw in enumerate(f, 1):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue

        try:
            row = [x.strip() for x in next(csv.reader([raw]))]
        except csv.Error:
            bad_rows.append((line_no, "CSV parse error", s))
            continue

        if row == HEADER:
            headers += 1
            continue

        if len(row) != 10:
            notes.append((line_no, s))
            continue

        label = row[0]
        try:
            sid = int(row[1])
            # Actual data order is sample index, then timestamp.
            sample_idx = int(row[2])
            timestamp_ms = int(row[3])
            vals = [float(x) for x in row[4:10]]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("non-finite sensor value")
        except (ValueError, OverflowError) as e:
            bad_rows.append((line_no, str(e), s))
            continue

        numeric_rows += 1
        labels[label] += 1
        key = tuple(row)
        if key in seen:
            duplicates += 1
        seen.add(key)

        windows[(label, sid)].append(
            (line_no, sample_idx, timestamp_ms, vals)
        )

        for channel, value in zip(CHANNELS, vals):
            limit = 2.0 if channel.startswith("a") else 1000.0
            if abs(value) > limit:
                range_violations.append(
                    (line_no, label, sid, sample_idx, channel, value)
                )
            if channel.startswith("a") and abs(value) >= 1.999:
                boundary_hits.append(
                    (line_no, label, sid, sample_idx, channel, value)
                )

print("=== SUMMARY ===")
print("Repeated headers:", headers)
print("Valid numeric rows:", numeric_rows)
print("Unique windows:", len(windows))
print("Non-CSV notes:", len(notes))
print("Invalid numeric rows:", len(bad_rows))
print("Exact duplicate rows:", duplicates)
print("Unexpected labels:", sorted(set(labels) - EXPECTED))
print("Missing labels:", sorted(EXPECTED - set(labels)))

print("\n=== COUNTS ===")
for label in sorted(EXPECTED):
    ids = sorted(sid for lab, sid in windows if lab == label)
    print(f"{label}: {len(ids)} windows; IDs={ids}")

print("\n=== WINDOW INTEGRITY ===")
issues = 0
for (label, sid), rows in sorted(windows.items()):
    indices = [r[1] for r in rows]
    if len(rows) != 50 or indices != list(range(50)):
        issues += 1
        print(
            f"ISSUE {label} id={sid}: rows={len(rows)}, "
            f"indices={indices}"
        )
print("Windows with row-count/index issues:", issues)

print("\n=== TIMESTAMP EXAMPLES ===")
for key in sorted(windows)[:3]:
    rows = windows[key]
    print(
        key,
        "first:", rows[0][2],
        "second:", rows[1][2],
        "last:", rows[-1][2],
    )

print("\n=== SENSOR RANGE ===")
print("Beyond configured ranges:", len(range_violations))
for item in range_violations[:20]:
    print(item)

print("\n=== ACCELEROMETER BOUNDARY HITS (REVIEW) ===")
print("Values with |a| >= 1.999g:", len(boundary_hits))
for item in boundary_hits[:30]:
    print(item)

print("\n=== NON-CSV NOTES ===")
for item in notes:
    print(item)

print("\n=== INVALID ROWS ===")
for item in bad_rows[:20]:
    print(item)
