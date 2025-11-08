#!/usr/bin/env python3
"""Test with the caching workaround"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from client import DimRedAPIClient

# Initialize client
api_key = "sk_test_v1_0D223I5q_pbQc2O8QB6r9S9WakV7X5iup4pXanAz5Sa"
client = DimRedAPIClient(api_key, "https://www.dimred.com")

# Create a new project and prompt
print("Creating a new prompt to test caching...")
project_id = client.create_project("Cache Test", "Testing prompt caching")

prompt_text = "You are a helpful assistant. Answer questions clearly and concisely."
output_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"]
}

prompt_id = client.create_prompt(
    project_id=project_id,
    prompt_text=prompt_text,
    prompt_message_type="system",
    name="Test Prompt",
    output_schema=output_schema
)

print(f"\nCreated prompt: {prompt_id}")

# Now try to fetch it
print("\nFetching prompt with client.get_prompt()...")
prompt_data = client.get_prompt(prompt_id)

if prompt_data.get('prompt_text'):
    print("✓ Successfully retrieved prompt from cache!")
    print(f"  Prompt text: {prompt_data['prompt_text'][:50]}...")
    print(f"  All fields: {list(prompt_data.keys())}")
else:
    print("✗ Could not retrieve prompt text")
    print(f"  Response: {prompt_data}")

# Try to fetch a prompt we didn't create (not in cache)
old_prompt_id = "9a5108a7-d91b-4d4a-8385-581bd5f69ce6"
print(f"\nFetching old prompt (not in cache): {old_prompt_id}")
old_prompt_data = client.get_prompt(old_prompt_id)

if old_prompt_data.get('prompt_text'):
    print("✓ Retrieved prompt text (API must be working now!)")
    print(f"  Prompt text: {old_prompt_data['prompt_text'][:50]}...")
else:
    print("✗ No prompt text available (expected - API returns HTML)")
    print(f"  Response: {old_prompt_data}")