#!/usr/bin/env python3
"""
DimRed API Inference Mode Example

This script demonstrates running inference mode with the DimRed API:
- Sets up a project, dataset, and prompt
- Runs inference (no metrics evaluation) on all datapoints
- Outputs the LLM responses without scoring

Usage:
    python inference.py --api-key YOUR_API_KEY [--base-url URL]
"""

import argparse
import json
import logging
import os
import sys

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
        description="DimRed API - Inference mode example"
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
            project_name="Inference Mode Test",
            project_description="Testing inference mode with DimRed API"
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
            name="Financial Crime Detection - Inference",
            output_schema=output_schema
        )

        # 5. Run inference mode (no metric needed)
        logger.info("\n=== Step 5: Run Inference Mode ===")
        logger.info("Running inference without metric evaluation...")

        result = client.run_workflow(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            model_name="gpt-4o-mini",
            provider="openai",
            mode="inference"
        )

        # Display results
        logger.info("\n=== Inference Results ===")
        if result.get("status"):
            logger.info(f"Status: {result.get('status')}")

        if result.get("task_id"):
            logger.info(f"Task ID: {result.get('task_id')}")

        if result.get("eval_id"):
            logger.info(f"Evaluation ID: {result.get('eval_id')}")

        # If the response includes outputs, display them
        if result.get("outputs"):
            logger.info(f"\nGenerated {len(result.get('outputs'))} outputs")
            for idx, output in enumerate(result.get("outputs")[:3], 1):  # Show first 3
                logger.info(f"\nOutput {idx}:")
                logger.info(json.dumps(output, indent=2))

        logger.info("\n✓ Inference mode completed successfully!")
        logger.info("Note: No metrics were evaluated in inference mode")

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