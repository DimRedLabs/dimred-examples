#!/usr/bin/env python3
"""Test different header combinations to get JSON response"""

import requests
import json

api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
prompt_id = "9a5108a7-d91b-4d4a-8385-581bd5f69ce6"
url = f"https://www.dimred.com/api/v1/prompts/{prompt_id}"

# Test different header combinations
header_sets = [
    {
        "name": "X-API-Key with Accept JSON",
        "headers": {
            "X-API-Key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    },
    {
        "name": "Authorization Bearer",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    },
    {
        "name": "X-API-Key only",
        "headers": {
            "X-API-Key": api_key
        }
    }
]

for header_set in header_sets:
    print(f"\nTesting: {header_set['name']}")
    print("-" * 40)

    response = requests.get(url, headers=header_set['headers'])
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")

    if 'application/json' in response.headers.get('content-type', ''):
        print("✓ Got JSON response!")
        data = response.json()
        print(f"  Keys: {list(data.keys())[:5]}...")
        if 'prompt_text' in data:
            print(f"  Has prompt_text: YES")
        break
    else:
        print("✗ Still HTML")