#!/usr/bin/env python3
"""Test dataset export endpoint"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from client import DimRedAPIClient

# Initialize client
api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
client = DimRedAPIClient(api_key, "https://www.dimred.com")

# Use the dataset from the just-run inference
dataset_id = "239e2e5c-1792-4967-82c1-7502d1827c86"

print(f"Testing export for dataset: {dataset_id}")
print("-" * 50)

# Try exporting
export_data = client.export_dataset(dataset_id)

print(f"\nExport response keys: {list(export_data.keys())}")
print(f"Total datapoints: {export_data.get('total_datapoints', 0)}")
print(f"Datapoints array length: {len(export_data.get('datapoints', []))}")

if export_data.get('datapoints'):
    dp = export_data['datapoints'][0]
    print(f"\nFirst datapoint keys: {list(dp.keys())}")
    if dp.get('output_data'):
        print(f"Has output_data: YES")
        print(f"Output preview: {str(dp['output_data'])[:100]}...")
    else:
        print(f"Has output_data: NO")

print("\nFull export response:")
print(json.dumps(export_data, indent=2)[:1000])