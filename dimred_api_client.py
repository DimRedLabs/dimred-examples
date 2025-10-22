#!/usr/bin/env python3
"""
DimRed API Client

A Python client for interacting with the DimRed backend API using API key authentication.
This script demonstrates the complete workflow:
1. Create a project
2. Create a dataset
3. Add data points to the dataset
4. Create a prompt
5. Create a metric
6. Run prompt tuning
7. Poll for tuning session completion

Usage:
    python dimred_api_client.py --api-key YOUR_API_KEY [--base-url URL]
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DimRedAPIError(Exception):
    """Custom exception for DimRed API errors"""
    pass


class DimRedAPIClient:
    """Client for interacting with DimRed API"""

    def __init__(self, api_key: str, base_url: str = "https://api.dimred.com"):
        """
        Initialize the DimRed API client.

        Args:
            api_key: DimRed API key for authentication
            base_url: Base URL for the API (default: https://api.dimred.com)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        logger.info(f"Initialized DimRed API client with base URL: {self.base_url}")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            DimRedAPIError: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"{method} {url}")
        if data:
            logger.debug(f"Request body: {json.dumps(data, indent=2)}")

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                params=params,
                timeout=30
            )

            logger.debug(f"Response status: {response.status_code}")

            # Try to parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                response_data = {"text": response.text}

            # Check for errors
            if response.status_code >= 400:
                error_detail = response_data.get("detail", response_data.get("message", "Unknown error"))
                raise DimRedAPIError(
                    f"API request failed with status {response.status_code}: {error_detail}"
                )

            return response_data

        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise DimRedAPIError(f"Request failed: {str(e)}")

    def create_project(self, project_name: str, project_description: str = "") -> str:
        """
        Create a new project.

        Args:
            project_name: Name of the project
            project_description: Optional description

        Returns:
            project_id: ID of the created project
        """
        logger.info(f"Creating project: {project_name}")

        data = {
            "project_name": project_name,
            "project_description": project_description
        }

        response = self._make_request("POST", "/api/v2/projects", data=data)
        project_id = response.get("project_id")

        if not project_id:
            raise DimRedAPIError("No project_id in response")

        logger.info(f"✓ Project created: {project_id}")
        return project_id

    def create_dataset(
        self,
        project_id: str,
        dataset_name: str,
        dataset_source: str = "api"
    ) -> str:
        """
        Create a new dataset.

        Args:
            project_id: ID of the project
            dataset_name: Name of the dataset
            dataset_source: Source of the dataset (default: "api")

        Returns:
            dataset_id: ID of the created dataset
        """
        logger.info(f"Creating dataset: {dataset_name} for project {project_id}")

        data = {
            "project_id": project_id,
            "dataset_name": dataset_name,
            "dataset_source": dataset_source
        }

        response = self._make_request("POST", "/api/v2/datasets", data=data)

        # Handle success response format
        if response.get("status") == "success":
            dataset_id = response.get("data", {}).get("dataset_id")
        else:
            dataset_id = response.get("dataset_id")

        if not dataset_id:
            raise DimRedAPIError("No dataset_id in response")

        logger.info(f"✓ Dataset created: {dataset_id}")
        return dataset_id

    def add_datapoints(self, dataset_id: str, datapoints: List[Dict[str, Any]]) -> int:
        """
        Add datapoints to a dataset.

        Args:
            dataset_id: ID of the dataset
            datapoints: List of datapoint dictionaries with keys:
                - input_data: Input data (string or JSON)
                - expected_output: Expected output (string or JSON)
                - output_data: Optional output data
                - data_metadata: Optional metadata

        Returns:
            added_count: Number of datapoints added
        """
        logger.info(f"Adding {len(datapoints)} datapoints to dataset {dataset_id}")

        # Send datapoints directly as the body (list, not wrapped in dict)
        response = self._make_request(
            "POST",
            f"/api/v2/datasets/{dataset_id}/datapoints",
            data=datapoints
        )

        # Handle success response format
        if response.get("status") == "success":
            added_count = response.get("data", {}).get("added_count", 0)
        else:
            added_count = response.get("added_count", 0)

        logger.info(f"✓ Added {added_count} datapoints")
        return added_count

    def create_prompt(
        self,
        project_id: str,
        messages: List[Dict[str, str]],
        name: str = "",
        output_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new prompt.

        Args:
            project_id: ID of the project
            messages: List of message dictionaries with keys:
                - prompt_text: Message text
                - prompt_message_type: Type (system, user, assistant)
            name: Optional prompt name
            output_schema: Optional JSON schema for structured output

        Returns:
            prompt_id: ID of the created prompt
        """
        logger.info(f"Creating prompt for project {project_id}")

        data = {
            "project_id": project_id,
            "messages": messages
        }

        if name:
            data["name"] = name
        if output_schema:
            data["output_schema"] = output_schema

        response = self._make_request("POST", "/api/v2/prompts", data=data)
        prompt_id = response.get("prompt_id")

        if not prompt_id:
            raise DimRedAPIError("No prompt_id in response")

        logger.info(f"✓ Prompt created: {prompt_id}")
        return prompt_id

    def create_metric(
        self,
        project_id: str,
        code: str,
        metric_name: str = "",
        metric_description: str = ""
    ) -> str:
        """
        Create a new metric from Python code.

        Args:
            project_id: ID of the project
            code: Python code defining metric_func(output, expected) -> float
            metric_name: Optional metric name (auto-generated if not provided)
            metric_description: Optional description (auto-generated if not provided)

        Returns:
            metric_id: ID of the created metric
        """
        logger.info(f"Creating metric for project {project_id}")

        data = {
            "project_id": project_id,
            "code": code
        }

        if metric_name:
            data["metric_name"] = metric_name
        if metric_description:
            data["metric_description"] = metric_description

        response = self._make_request("POST", "/api/v2/metrics/pkl-metrics/code", data=data)
        metric_id = response.get("metric_id")

        if not metric_id:
            raise DimRedAPIError("No metric_id in response")

        logger.info(f"✓ Metric created: {metric_id}")
        if response.get("metadata_auto_generated"):
            logger.info(f"  Auto-generated name: {response.get('metric_name')}")

        return metric_id

    def run_tuning(
        self,
        project_id: str,
        dataset_id: str,
        prompt_id: str,
        metric_id: str,
        num_iterations: int = 3,
        include_project_metrics: bool = False,
        model_name: str = "gpt-4o-mini",
        provider: str = "openai"
    ) -> Dict[str, str]:
        """
        Start a prompt tuning session.

        Args:
            project_id: ID of the project
            dataset_id: ID of the dataset
            prompt_id: ID of the prompt
            metric_id: ID of the target metric
            num_iterations: Number of tuning iterations (1-10)
            include_project_metrics: Whether to include project metrics
            model_name: LLM model name
            provider: LLM provider (openai, anthropic, openrouter)

        Returns:
            Dictionary with tuning_session_id and task_id
        """
        logger.info(f"Starting tuning session for project {project_id}")
        logger.info(f"  Dataset: {dataset_id}")
        logger.info(f"  Prompt: {prompt_id}")
        logger.info(f"  Metric: {metric_id}")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Model: {model_name} ({provider})")

        data = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "prompt_id": prompt_id,
            "metric_id": metric_id,
            "include_project_metrics": include_project_metrics,
            "num_iterations": num_iterations,
            "model_settings": {
                "model_name": model_name,
                "provider": provider
            }
        }

        response = self._make_request("POST", "/api/v2/prompts/tune", data=data)

        tuning_session_id = response.get("tuning_session_id")
        task_id = response.get("task_id")

        if not tuning_session_id or not task_id:
            raise DimRedAPIError("Missing tuning_session_id or task_id in response")

        logger.info(f"✓ Tuning started")
        logger.info(f"  Session ID: {tuning_session_id}")
        logger.info(f"  Task ID: {task_id}")

        return {
            "tuning_session_id": tuning_session_id,
            "task_id": task_id,
            "status": response.get("status", "pending")
        }

    def get_tuning_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current status of a tuning session.

        Args:
            session_id: ID of the tuning session

        Returns:
            Dictionary with tuning session details
        """
        response = self._make_request(
            "GET",
            f"/api/v2/prompts/tune/{session_id}"
        )
        return response

    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get a specific prompt by ID.

        Args:
            prompt_id: ID of the prompt

        Returns:
            Dictionary with prompt details
        """
        response = self._make_request(
            "GET",
            f"/api/v2/prompts/{prompt_id}"
        )
        return response

    def get_best_prompt(self, session_id: str) -> Dict[str, Any]:
        """
        Get the best prompt from a tuning session.

        Args:
            session_id: ID of the tuning session

        Returns:
            Dictionary with best prompt details including:
                - prompt_id: ID of the best prompt
                - messages: List of prompt messages
                - metrics: Performance metrics
                - iteration: Which iteration this was from
        """
        logger.info(f"Fetching best prompt for session {session_id}")

        # Get the tuning session
        session = self.get_tuning_session(session_id)

        prompt_id = session.get('prompt_id')
        if not prompt_id:
            raise DimRedAPIError("No prompt_id found in tuning session")

        # Fetch the prompt details
        prompt_response = self.get_prompt(prompt_id)

        # Combine session metrics with prompt details
        result = {
            "prompt_id": prompt_id,
            "messages": prompt_response.get("messages", []),
            "metrics": session.get("metrics"),
            "iteration": session.get("iteration"),
            "is_best": session.get("is_best", False),
            "eval_id": session.get("eval_id")
        }

        logger.info(f"✓ Retrieved best prompt: {prompt_id}")
        return result

    def wait_for_tuning_completion(
        self,
        session_id: str,
        poll_interval: int = 15,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Poll a tuning session until it completes or times out.

        Args:
            session_id: ID of the tuning session
            poll_interval: Seconds between polls (default: 15)
            timeout: Maximum seconds to wait (default: 3600 = 1 hour)

        Returns:
            Final tuning session data

        Raises:
            DimRedAPIError: If timeout reached or tuning failed
        """
        logger.info(f"Polling tuning session {session_id} every {poll_interval}s")
        logger.info(f"Timeout: {timeout}s")

        start_time = time.time()
        last_status = None

        poll_count = 0
        while True:
            elapsed = time.time() - start_time
            poll_count += 1

            if elapsed > timeout:
                raise DimRedAPIError(f"Timeout reached after {timeout}s")

            # Get session status
            logger.debug(f"Poll #{poll_count}: Fetching session status (elapsed: {elapsed:.1f}s)")
            session = self.get_tuning_session(session_id)
            status = session.get("status", "unknown")

            # Log status changes
            if status != last_status:
                logger.info(f"Status: {status}")
                last_status = status
            else:
                logger.debug(f"Poll #{poll_count}: Status unchanged: {status}")

            # Check if completed
            if status == "completed":
                logger.info("✓ Tuning completed successfully")

                # Extract best iteration info
                prompt_id = session.get("prompt_id")
                eval_id = session.get("eval_id")
                is_best = session.get("is_best", False)

                logger.info(f"  Best prompt ID: {prompt_id}")
                logger.info(f"  Best eval ID: {eval_id}")
                logger.info(f"  Is best iteration: {is_best}")

                return session

            # Check if failed
            if status == "failed":
                error_msg = session.get("error_message", "Unknown error")
                raise DimRedAPIError(f"Tuning failed: {error_msg}")

            # Check if cancelled
            if status == "cancelled":
                raise DimRedAPIError("Tuning was cancelled")

            # Wait before next poll
            logger.debug(f"Sleeping for {poll_interval}s before next poll")
            time.sleep(poll_interval)


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="DimRed API Client - Complete workflow demonstration"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="DimRed API key"
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

        # 3. Add datapoints from example_data (8).json
        logger.info("\n=== Step 3: Add Datapoints ===")

        # Load data from file
        data_file_path = "example.json"
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
        tuning_result = client.run_tuning(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            metric_id=metric_id,
            num_iterations=3,
            model_name="gpt-4.1-mini",
            provider="openai"
        )

        session_id = tuning_result["tuning_session_id"]

        # 7. Wait for completion
        logger.info("\n=== Step 7: Wait for Completion ===")
        final_result = client.wait_for_tuning_completion(
            session_id=session_id,
            poll_interval=15,
            timeout=3600
        )

        # Fetch the best prompt
        prompt_id = final_result.get('prompt_id')
        prompt_text = None
        if prompt_id:
            logger.info("\n=== Fetching Best Prompt ===")
            try:
                prompt_response = client.get_prompt(prompt_id)
                # Extract prompt text from messages array
                if prompt_response and "messages" in prompt_response and len(prompt_response["messages"]) > 0:
                    prompt_text = prompt_response["messages"][0].get("prompt_text")
            except Exception as e:
                logger.warning(f"Failed to fetch best prompt: {e}")
                prompt_text = None

        # Print final results
        logger.info("\n=== Final Results ===")
        logger.info(f"Session ID: {final_result['session_id']}")
        logger.info(f"Status: {final_result['status']}")
        logger.info(f"Best Prompt ID: {final_result.get('prompt_id', 'N/A')}")
        logger.info(f"Best Eval ID: {final_result.get('eval_id', 'N/A')}")
        logger.info(f"Iteration: {final_result.get('iteration', 'N/A')}")
        logger.info(f"Is Best: {final_result.get('is_best', 'N/A')}")

        if final_result.get("metrics"):
            logger.info("\nMetrics:")
            logger.info(json.dumps(final_result["metrics"], indent=2))

        # Display the best prompt text
        if prompt_text:
            logger.info("\n=== Best Prompt ===")
            logger.info(prompt_text)
        else:
            logger.info("\nBest Prompt: N/A")

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