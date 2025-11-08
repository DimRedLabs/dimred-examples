#!/usr/bin/env python3
"""
DimRed API Tuning Example

This script demonstrates the complete workflow for using the DimRed API:
1. Create a project
2. Create a dataset
3. Add data points to the dataset
4. Create a prompt
5. Create a metric
6. Run prompt tuning
7. Poll for tuning session completion

Usage:
    python run_tuning.py --api-key YOUR_API_KEY [--base-url URL]
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
        description="DimRed API Client - Complete workflow demonstration"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DIMRED_API_KEY"),
        help="DimRed API key (or set DIMRED_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        default="https://www.dimred.com",
        help="API base URL (default: https://www.dimred.com)"
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
            project_name="API Test Project",
            project_description="Testing DimRed API with Python client"
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
        prompt_text = (
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
        )

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
            prompt_text=prompt_text,
            prompt_message_type="system",
            name="Financial Crime Perpetrator Detection",
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

        # 6. Run tuning
        logger.info("\n=== Step 6: Run Tuning ===")
        workflow_response = client.run_workflow(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            metric_id=metric_id,
            model_name="gpt-4o-mini",
            provider="openai",
            mode="tune",
            num_iterations=3
        )

        workflow_id = workflow_response.get("id") or workflow_response.get("tuning_session_id")
        logger.info(f"Tuning session started: {workflow_id}")

        # 7. Wait for completion
        logger.info("\n=== Step 7: Wait for Completion ===")
        final_result = client.wait_for_workflow_completion(
            workflow_id=workflow_id,
            poll_interval=10,
            timeout=3600
        )

        # Extract results from the workflow
        metrics_data = final_result.get("metrics", {})
        best_prompt_id = metrics_data.get("best_prompt_id")
        final_prompt_id = metrics_data.get("final_prompt_id")

        # Print final results
        logger.info("\n=== Final Results ===")
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Status: {final_result.get('status')}")
        logger.info(f"Best Prompt ID: {best_prompt_id or 'N/A'}")
        logger.info(f"Final Prompt ID: {final_prompt_id or 'N/A'}")
        logger.info(f"Reason: {metrics_data.get('reason', 'N/A')}")

        # Display iteration results
        if metrics_data:
            final_metrics = metrics_data.get("final_metrics", {})
            if final_metrics:
                logger.info("\nFinal Metrics:")
                for metric_name, value in final_metrics.items():
                    logger.info(f"  {metric_name}: {value}")

            iteration_results = metrics_data.get("iteration_results", [])
            if iteration_results:
                logger.info(f"\nCompleted {len(iteration_results)} iterations")
                for result in iteration_results:
                    iter_num = result.get("iteration", "?")
                    iter_metrics = result.get("metrics", {})
                    logger.info(f"  Iteration {iter_num}: {iter_metrics}")

        # Step 8: Best Prompt Information
        logger.info("\n=== Step 8: Best Prompt Information ===")
        best_prompt_id = metrics_data.get('best_prompt_id')
        final_prompt_id = metrics_data.get('final_prompt_id')

        if best_prompt_id:
            logger.info(f"Best Prompt ID: {best_prompt_id}")

            if final_prompt_id and final_prompt_id != best_prompt_id:
                logger.info(f"Final Prompt ID: {final_prompt_id}")

            # Check if optimization occurred
            if prompt_id == best_prompt_id:
                logger.info("\n✓ The original prompt was already optimal for this dataset!")
            else:
                logger.info("\n✓ A new optimized prompt was generated through tuning!")
                logger.info("  The tuning process tested variations and found improvements.")

            logger.info("\n📝 Use this prompt ID in future workflows for best performance:")
            logger.info(f"   client.run_workflow(..., prompt_id='{best_prompt_id}', ...)")
        else:
            logger.info("Best prompt ID not available in results")

        logger.info("\n✓ All steps completed successfully!")

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
