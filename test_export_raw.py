#!/usr/bin/env python3
"""Test dataset export endpoint raw response"""

import requests

api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
dataset_id = "239e2e5c-1792-4967-82c1-7502d1827c86"

headers = {"X-API-Key": api_key}
url = f"https://www.dimred.com/api/v1/datasets/{dataset_id}/export"

print(f"Testing GET {url}")
print("-" * 50)

response = requests.get(url, headers=headers)
print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")

if 'application/json' in response.headers.get('content-type', ''):
    try:
        data = response.json()
        print(f"\nResponse is JSON!")
        print(f"Keys: {list(data.keys())}")
        print(f"Total datapoints: {data.get('total_datapoints', 'Not found')}")

        if data.get('datapoints'):
            print(f"Number of datapoints: {len(data['datapoints'])}")
        else:
            print("No datapoints in response")

    except Exception as e:
        print(f"Failed to parse JSON: {e}")
elif 'text/html' in response.headers.get('content-type', ''):
    print("\n❌ Response is HTML (not JSON)")
    print("First 500 chars:")
    print(response.text[:500])
else:
    print(f"\nResponse text (first 500 chars):")
    print(response.text[:500])