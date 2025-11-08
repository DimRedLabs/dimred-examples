#!/usr/bin/env python3
"""Test to see the raw output of client.get_prompt()"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from client import DimRedAPIClient

# Initialize client
api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
client = DimRedAPIClient(api_key, "https://www.dimred.com")

# Use a known prompt ID from earlier test
prompt_id = "9a5108a7-d91b-4d4a-8385-581bd5f69ce6"

print(f"Calling client.get_prompt('{prompt_id}')")
print("-" * 50)

result = client.get_prompt(prompt_id)

print(f"Type: {type(result)}")
print(f"Result: {result}")
print()

if isinstance(result, dict):
    print("Keys in result:", list(result.keys()) if result else "Empty dict")

    if result:
        print("\nFull result structure:")
        print(json.dumps(result, indent=2))
    else:
        print("\nResult is an empty dictionary {}")