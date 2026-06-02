import json
import os
import sys

# Enforce UTF-8 output on Windows terminal
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

json_path = "layer6_layer8_spikes_truth_table.json"
if not os.path.exists(json_path):
    print(f"[Error] File not found: {json_path}")
    exit(1)

print(f"Reading {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

print("Filtering spikes to only keep layer8 label '7'...")
for rec in records:
    # Remove all layer8 spikes except label '7'
    for lbl in ["6", "5", "4", "3", "2", "1", "0"]:
        if lbl in rec["layer8_spikes"]:
            del rec["layer8_spikes"][lbl]

print(f"Writing filtered records back to {json_path}...")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=4)

print("Successfully deleted specified neurons and kept only label '7'!")
