import json
import os

json_path = "layer6_layer8_spikes_truth_table.json"
if not os.path.exists(json_path):
    print(f"[Error] File not found: {json_path}")
    exit(1)

print(f"Reading {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

print("Filtering spikes...")
for rec in records:
    # Remove p2, p1, p0 from layer6_spikes
    for lbl in ["p2", "p1", "p0"]:
        if lbl in rec["layer6_spikes"]:
            del rec["layer6_spikes"][lbl]
            
    # Remove 3, 2, 1, 0 from layer8_spikes
    for lbl in ["3", "2", "1", "0"]:
        if lbl in rec["layer8_spikes"]:
            del rec["layer8_spikes"][lbl]

print(f"Writing filtered records back to {json_path}...")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=4)

import sys
# Enforce UTF-8 output on Windows terminal
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("Successfully deleted specified neurons from the spikes truth table JSON file!")
