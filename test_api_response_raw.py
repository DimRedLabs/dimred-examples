#!/usr/bin/env python3
"""Test to see the actual API response for get_prompt"""

import requests

# Direct API call
api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
prompt_id = "9a5108a7-d91b-4d4a-8385-581bd5f69ce6"
url = f"https://www.dimred.com/api/v1/prompts/{prompt_id}"

headers = {
    "X-API-Key": api_key,
    "Accept": "application/json",
    "Content-Type": "application/json"
}

print(f"Making direct API call to: {url}")
print("-" * 50)

response = requests.get(url, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Content Type: {response.headers.get('content-type')}")
print()

# Try to parse as JSON
try:
    json_data = response.json()
    print("Successfully parsed as JSON:")
    import json
    print(json.dumps(json_data, indent=2))
except:
    print("Could not parse as JSON")
    print(f"Response text (first 500 chars):")
    print(response.text[:500])