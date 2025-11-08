#!/usr/bin/env python3
"""Test the exact function provided"""

def test_get_prompt(prompt_id, api_key):
    import requests

    headers = {"X-API-Key": api_key}
    url = f"https://www.dimred.com/api/v1/prompts/{prompt_id}"

    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")

    data = response.json()
    print(f"Response keys: {data.keys()}")

    if "prompt_text" in data:
        print(f"Prompt text: {data['prompt_text']}")
    else:
        print("WARNING: prompt_text not found in response!")
        print(f"Full response: {data}")

    return data

# Test it
api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
prompt_id = "9a5108a7-d91b-4d4a-8385-581bd5f69ce6"

print("Testing the exact function provided:")
print("=" * 50)
try:
    result = test_get_prompt(prompt_id, api_key)
except Exception as e:
    print(f"ERROR: {e}")
    print("\nThe response is not JSON - it's HTML!")

    # Show what we actually get
    import requests
    headers = {"X-API-Key": api_key}
    url = f"https://www.dimred.com/api/v1/prompts/{prompt_id}"
    response = requests.get(url, headers=headers)
    print(f"\nActual response content-type: {response.headers.get('content-type')}")
    print(f"First 200 chars: {response.text[:200]}...")