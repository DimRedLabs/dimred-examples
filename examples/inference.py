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
            name="Financial Crime Detection - Inference",
            output_schema=output_schema
        )

        # 5. Create a dummy metric for inference (required by API)
        logger.info("\n=== Step 5: Create Dummy Metric ===")
        logger.info("Creating dummy metric for inference mode...")

        try:
            # Create a simple always-pass metric for inference
            metric_code = '''
def metric_func(output, expected):
    """Dummy metric that always returns 1.0 for inference mode"""
    return 1.0
'''
            metric_id = client.create_metric(
                project_id=project_id,
                code=metric_code,
                metric_name="Inference Dummy Metric",
                metric_description="Placeholder metric for inference mode (always returns 1.0)"
            )
            logger.info(f"✓ Dummy metric created: {metric_id}")
        except DimRedAPIError as e:
            logger.warning(f"Could not create dummy metric: {e}")
            metric_id = None

        # 6. Run inference mode and wait for completion
        logger.info("\n=== Step 6: Run Inference Mode ===")
        logger.info("Running inference without metric evaluation...")

        # Start workflow with the metric (now using evaluate mode since we have a metric)
        workflow_response = client.run_workflow(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            metric_id=metric_id,  # Use the dummy metric
            model_name="gpt-4o-mini",
            provider="openai",
            mode="evaluate" if metric_id else "inference"  # Use evaluate mode when we have a metric
        )

        workflow_id = workflow_response.get("id")
        if workflow_id:
            logger.info(f"Workflow started with ID: {workflow_id}")
            logger.info(f"Initial status: {workflow_response.get('status')}")

            # Wait for completion
            logger.info("\n=== Step 7: Monitor Workflow Progress ===")
            try:
                completed_workflow = client.wait_for_workflow_completion(
                    workflow_id=workflow_id,
                    poll_interval=3,  # Poll every 3 seconds for inference
                    timeout=300  # 5 minute timeout
                )

                # Get detailed results
                logger.info("\n=== Step 8: Display Inference Summary ===")

                # Note: The dataset export endpoint currently returns HTML instead of JSON,
                # so we can't fetch the actual inference outputs yet.
                # Display what we know from the workflow completion.

                logger.info("Inference workflow completed successfully!")
                logger.info(f"  Workflow ID: {workflow_id}")
                logger.info(f"  Dataset ID: {dataset_id}")
                logger.info(f"  Prompt ID: {prompt_id}")

                # Try to get metrics from completed workflow
                if completed_workflow.get('metrics'):
                    logger.info("\nWorkflow Metrics:")
                    metrics = completed_workflow.get('metrics', {})
                    for key, value in metrics.items():
                        logger.info(f"  {key}: {value}")

                # Try dataset export anyway in case it starts working
                try:
                    export_data = client.export_dataset(dataset_id)
                    total_datapoints = export_data.get('total_datapoints', 0)

                    if total_datapoints > 0:
                        logger.info(f"\n✓ Retrieved {total_datapoints} datapoints with inference results")

                    # Display the inference results
                    datapoints = export_data.get("datapoints", [])

                    if datapoints:
                        logger.info(f"\n=== Inference Outputs ===")

                        # Show first 5 results in detail
                        for idx, dp in enumerate(datapoints[:5], 1):
                            logger.info(f"\n--- Result {idx} ---")

                            # Parse input data
                            input_data = dp.get("input_data")
                            if input_data:
                                try:
                                    input_obj = json.loads(input_data) if isinstance(input_data, str) else input_data
                                    logger.info(f"Input: {json.dumps(input_obj, indent=2)}")
                                except:
                                    logger.info(f"Input: {input_data}")

                            # Parse output data (the inference result)
                            output_data = dp.get("output_data")
                            if output_data:
                                try:
                                    output_obj = json.loads(output_data) if isinstance(output_data, str) else output_data
                                    logger.info(f"LLM Output: {json.dumps(output_obj, indent=2)}")
                                except:
                                    logger.info(f"LLM Output: {output_data}")
                            else:
                                logger.info("LLM Output: (not yet available)")

                            # Show expected output for comparison
                            expected_output = dp.get("expected_output")
                            if expected_output:
                                try:
                                    expected_obj = json.loads(expected_output) if isinstance(expected_output, str) else expected_output
                                    logger.info(f"Expected: {json.dumps(expected_obj, indent=2)}")
                                except:
                                    logger.info(f"Expected: {expected_output}")

                        if len(datapoints) > 5:
                            logger.info(f"\n... and {len(datapoints) - 5} more results")

                        # Count how many have output_data populated
                        with_outputs = sum(1 for dp in datapoints if dp.get("output_data"))
                        logger.info(f"\n✓ {with_outputs}/{len(datapoints)} datapoints have inference outputs")
                    else:
                        logger.info("\nNote: Dataset export API endpoint is not yet returning JSON data.")
                        logger.info("The inference was completed but results cannot be fetched via the export endpoint yet.")

                except Exception as e:
                    logger.warning(f"Could not fetch dataset export: {e}")
                    logger.info("Falling back to workflow results...")

                    # Fallback to original method
                    result_mode = "evaluate" if metric_id else "inference"
                    results = client.get_workflow_results(workflow_id, result_mode)

                    if results.get("summary"):
                        logger.info("\nSummary:")
                        logger.info(json.dumps(results["summary"], indent=2))

                logger.info("\n✓ Inference mode completed successfully!")
                logger.info("Note: No metrics were evaluated in inference mode")

            except DimRedAPIError as e:
                logger.error(f"Workflow monitoring failed: {e}")
                # Try to cancel the workflow if it's still running
                try:
                    client.cancel_workflow(workflow_id)
                    logger.info("Workflow cancelled")
                except:
                    pass
                raise

        else:
            # Fallback to basic response display
            logger.info("\n=== Inference Started ===")
            logger.info(f"Status: {workflow_response.get('status')}")
            logger.info("Use workflow monitoring methods to track progress")
            logger.info("\n✓ Inference workflow submitted successfully!")

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