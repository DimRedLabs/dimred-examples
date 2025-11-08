#!/usr/bin/env python3
"""Test to see what the prompt API actually returns"""

import requests
import json

def test_get_prompt(prompt_id, api_key):
    headers = {"X-API-Key": api_key}
    url = f"https://www.dimred.com/api/v1/prompts/{prompt_id}"

    print(f"Testing GET {url}")
    print(f"Headers: {headers}")
    print("-" * 50)

    response = requests.get(url, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")

    # Check if it's JSON or HTML
    if 'application/json' in response.headers.get('content-type', ''):
        try:
            data = response.json()
            print(f"\nResponse is JSON!")
            print(f"Response keys: {list(data.keys())}")

            if "prompt_text" in data:
                print(f"\n✓ Found prompt_text field!")
                print(f"Prompt text preview: {data['prompt_text'][:100]}...")
            else:
                print("\n⚠️ WARNING: prompt_text not found in response!")

            print(f"\nFull response:")
            print(json.dumps(data, indent=2))

            return data
        except Exception as e:
            print(f"Failed to parse JSON: {e}")

    elif 'text/html' in response.headers.get('content-type', ''):
        print("\n❌ Response is HTML (not JSON)")
        print("First 500 chars of HTML:")
        print(response.text[:500])
    else:
        print(f"\nUnexpected content type")
        print(f"Response text: {response.text[:500]}")

    return None

# Test with a known prompt ID from our earlier tests
api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
prompt_id = "9a5108a7-d91b-4d4a-8385-581bd5f69ce6"

result = test_get_prompt(prompt_id, api_key)