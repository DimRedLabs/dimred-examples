#!/usr/bin/env python3
"""
DimRed API Tune Mode Example

This script demonstrates running tune mode with the DimRed API:
- Sets up a project, dataset, prompt, and metric
- Runs multiple tuning iterations with the new workflow API
- Waits for completion and shows the best prompt

Usage:
    python tune.py --api-key YOUR_API_KEY [--base-url URL] [--iterations N]
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from client import DimRedAPIClient, DimRedAPIError

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="DimRed API - Tune mode example"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DIMRED_API_KEY"),
        help="DimRed API key (or set DIMRED_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.dimred.com",
        help="API base URL (default: https://api.dimred.com)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of tuning iterations (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate API key
    if not args.api_key:
        logger.error("Error: No API key provided. Either pass --api-key or set DIMRED_API_KEY environment variable.")
        return 1

    try:
        # Initialize client
        client = DimRedAPIClient(args.api_key, args.base_url)

        # 1. Create project
        logger.info("\n=== Step 1: Create Project ===")
        project_id = client.create_project(
            project_name="Tune Mode Test",
            project_description="Testing tune mode with new workflow API"
        )

        # 2. Create dataset
        logger.info("\n=== Step 2: Create Dataset ===")
        dataset_id = client.create_dataset(
            project_id=project_id,
            dataset_name="Financial Crime Detection Dataset"
        )

        # 3. Add datapoints from example.json
        logger.info("\n=== Step 3: Add Datapoints ===")

        # Load data from file
        data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'example.json')
        with open(data_file_path, 'r') as f:
            example_data = json.load(f)

        logger.info(f"Loaded {len(example_data)} datapoints from {data_file_path}")

        # Convert to API format
        datapoints = []
        for item in example_data:
            datapoints.append({
                "input_data": json.dumps(item["input"]),
                "expected_output": json.dumps(item["expected"])
            })

        count = client.add_datapoints(dataset_id, datapoints)

        # 4. Create prompt for financial crime detection
        logger.info("\n=== Step 4: Create Prompt ===")
        messages = [
            {
                "prompt_text": (
                    "You are an expert financial crime analyst. Your task is to analyze news article "
                    "snippets and determine whether the person mentioned is a perpetrator of financial crime.\n\n"
                    "A person is a PERPETRATOR if:\n"
                    "- They are explicitly charged, indicted, arrested, or accused of financial crimes\n"
                    "- There is clear evidence of illegal activity (e.g., court documents, bank records)\n"
                    "- They are directly involved in illegal financial transactions\n\n"
                    "A person is NOT a perpetrator if:\n"
                    "- They are law enforcement, prosecutors, or investigators\n"
                    "- They are witnesses, victims, or observers\n"
                    "- There is only speculation or suspicion without charges\n"
                    "- They are community leaders or officials responding to crimes\n\n"
                    "Respond with JSON containing:\n"
                    "- is_perpetrator: true or false\n"
                    "- reasoning: brief explanation of your decision"
                ),
                "prompt_message_type": "system"
            }
        ]

        # Create output schema for structured JSON response
        output_schema = {
            "type": "object",
            "properties": {
                "is_perpetrator": {
                    "type": "boolean",
                    "description": "Whether the person is a perpetrator of financial crime"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the decision"
                }
            },
            "required": ["is_perpetrator", "reasoning"]
        }

        prompt_id = client.create_prompt(
            project_id=project_id,
            messages=messages,
            name="Financial Crime Detection - Initial",
            output_schema=output_schema
        )

        # 5. Create metric for perpetrator classification
        logger.info("\n=== Step 5: Create Metric ===")
        metric_code = '''
import json

def metric_func(output, expected):
    """
    Check if the LLM correctly identified whether someone is a perpetrator.
    Returns 1.0 for correct classification, 0.0 for incorrect.
    """
    # Parse output and expected if they're strings
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return 0.0

    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return 0.0

    # Extract is_perpetrator field
    output_value = output.get("is_perpetrator")
    expected_value = expected.get("is_perpetrator")

    # Both must be present and match
    if output_value is None or expected_value is None:
        return 0.0

    # Return 1.0 if they match, 0.0 if they don't
    return 1.0 if output_value == expected_value else 0.0
'''
        metric_id = client.create_metric(
            project_id=project_id,
            code=metric_code,
            metric_name="Perpetrator Classification Accuracy",
            metric_description="Measures whether the model correctly identifies perpetrators vs non-perpetrators"
        )

        # 6. Run tune mode (multiple iterations)
        logger.info("\n=== Step 6: Run Tune Mode ===")
        logger.info(f"Running tuning with {args.iterations} iterations...")

        result = client.run_workflow(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            metric_id=metric_id,
            model_name="gpt-4o-mini",
            provider="openai",
            mode="tune",
            num_iterations=args.iterations,
            include_project_metrics=True
        )

        # Get session ID for tracking
        session_id = result.get("tuning_session_id")

        if not session_id:
            logger.warning("No tuning_session_id returned. Checking immediate results...")

            # Display immediate results if available
            if result.get("status"):
                logger.info(f"Status: {result.get('status')}")
            if result.get("metrics"):
                logger.info("\nMetrics:")
                logger.info(json.dumps(result.get("metrics"), indent=2))

            logger.info("\n✓ Tune mode request completed")
            return 0

        # 7. Wait for completion and track progress
        logger.info(f"\n=== Step 7: Monitoring Tuning Session ===")
        logger.info(f"Session ID: {session_id}")

        # Poll for completion
        final_result = client.wait_for_tuning_completion(
            session_id=session_id,
            poll_interval=15,
            timeout=3600
        )

        # 8. Fetch and display the best prompt
        prompt_id = final_result.get('prompt_id')
        prompt_text = None
        if prompt_id:
            logger.info("\n=== Step 8: Fetching Best Prompt ===")
            try:
                prompt_response = client.get_prompt(prompt_id)
                # Extract prompt text from messages array
                if prompt_response and "messages" in prompt_response and len(prompt_response["messages"]) > 0:
                    prompt_text = prompt_response["messages"][0].get("prompt_text")
            except Exception as e:
                logger.warning(f"Failed to fetch best prompt: {e}")
                prompt_text = None

        # Display final results
        logger.info("\n=== Final Tuning Results ===")
        logger.info(f"Session ID: {final_result.get('session_id', 'N/A')}")
        logger.info(f"Status: {final_result.get('status', 'N/A')}")
        logger.info(f"Best Prompt ID: {final_result.get('prompt_id', 'N/A')}")
        logger.info(f"Best Eval ID: {final_result.get('eval_id', 'N/A')}")
        logger.info(f"Best Iteration: {final_result.get('iteration', 'N/A')}")

        if final_result.get("metrics"):
            logger.info("\nFinal Metrics:")
            logger.info(json.dumps(final_result["metrics"], indent=2))

        # Display the optimized prompt
        if prompt_text:
            logger.info("\n=== Optimized Prompt ===")
            logger.info("The following prompt achieved the best performance:")
            logger.info("-" * 60)
            logger.info(prompt_text)
            logger.info("-" * 60)
        else:
            logger.info("\nOptimized Prompt: N/A")

        logger.info("\n✓ Tune mode completed successfully!")
        logger.info(f"Completed {args.iterations} tuning iterations")

        return 0

    except DimRedAPIError as e:
        logger.error(f"\n✗ API Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\n\n✗ Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())