import json
import pandas as pd
import sys
import os

# Enforce UTF-8 output on Windows terminal
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

json_path = "layer6_layer8_spikes_truth_table.json"
if not os.path.exists(json_path):
    print(f"[Error] File not found: {json_path}")
    sys.exit(1)

print(f"Loading {json_path}...")
with open(json_path, 'r', encoding='utf-8') as f:
    records = json.load(f)

# Flatten the nested dictionary structures
rows = []
for rec in records:
    row = {
        "A": rec["A"],
        "B": rec["B"],
        "Expected_Product": rec["expected_product"],
        "Predicted_Product": rec["predicted_product"]
    }
    
    # Flatten Layer 6 spikes
    for k, v in rec["layer6_spikes"].items():
        row[f"L6_{k}"] = v
        
    # Flatten Layer 8 spikes
    for k, v in rec["layer8_spikes"].items():
        row[f"L8_{k}"] = v
        
    rows.append(row)

# Create pandas DataFrame
df = pd.DataFrame(rows)

# Save to CSV
csv_path = "layer6_layer8_spikes_flat_table.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")
print(f"Successfully generated flattened spikes table and saved to: {csv_path}")

# Print DataFrame details and head
print(f"\nDataFrame shape: {df.shape} (256 cases, {df.shape[1]} columns)")
print("\n📊 First 20 Rows Preview:")
print(df.head(20).to_string(index=False))
