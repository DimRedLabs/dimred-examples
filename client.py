"""
DimRed API Client

A Python client for interacting with the DimRed backend API using API key authentication.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Generator

import requests
try:
    import sseclient
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False

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

    def __init__(self, api_key: str, base_url: str = "https://www.dimred.com"):
        """
        Initialize the DimRed API client.

        Args:
            api_key: DimRed API key for authentication
            base_url: Base URL for the API (default: https://www.dimred.com)
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        # Cache for prompt details since GET prompt endpoint returns HTML
        self._prompt_cache = {}
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
                # Return empty dict if we can't parse JSON (e.g., HTML response)
                if response.text.startswith('<!DOCTYPE'):
                    return {}
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

        response = self._make_request("POST", "/api/v1/projects", data=data)

        # Log the response for debugging
        logger.debug(f"Project creation response: {response}")

        # Handle both direct project_id and nested in data field
        # The API returns 'id' not 'project_id'
        if response.get("data") and isinstance(response.get("data"), dict):
            project_id = response.get("data").get("id") or response.get("data").get("project_id")
        else:
            project_id = response.get("id") or response.get("project_id")

        if not project_id:
            logger.error(f"Unexpected response format: {response}")
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

        response = self._make_request("POST", "/api/v1/datasets", data=data)

        # Handle success response format
        # The API returns 'id' not 'dataset_id'
        if response.get("status") == "success":
            dataset_id = response.get("data", {}).get("id") or response.get("data", {}).get("dataset_id")
        else:
            dataset_id = response.get("id") or response.get("dataset_id")

        if not dataset_id:
            logger.error(f"Unexpected response format: {response}")
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
            f"/api/v1/datasets/{dataset_id}/datapoints",
            data=datapoints
        )

        # Handle success response format
        if response.get("status") == "success":
            added_count = response.get("data", {}).get("added_count", 0)
        else:
            added_count = response.get("added_count", 0)

        logger.info(f"✓ Added {added_count} datapoints")
        return added_count

    def export_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Export all datapoints from a dataset.
        After inference, the output_data field will contain the LLM outputs.

        Args:
            dataset_id: ID of the dataset to export

        Returns:
            Dictionary with:
                - dataset_id: The dataset ID
                - total_datapoints: Number of datapoints
                - datapoints: List of datapoint dictionaries with input_data,
                              expected_output, and output_data (if inference was run)
                - exported_at: Timestamp of export
        """
        logger.info(f"Exporting dataset {dataset_id}")

        response = self._make_request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/export"
        )

        total = response.get("total_datapoints", 0)
        logger.info(f"✓ Exported {total} datapoints")

        # Count how many have inference outputs
        datapoints = response.get("datapoints", [])
        with_outputs = sum(1 for dp in datapoints if dp.get("output_data"))
        if with_outputs > 0:
            logger.info(f"  {with_outputs} datapoints have inference outputs")

        return response

    def create_prompt(
        self,
        project_id: str,
        messages: List[Dict[str, str]] = None,
        prompt_text: str = None,
        prompt_message_type: str = "system",
        name: str = "",
        output_schema: Optional[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new prompt.

        Args:
            project_id: ID of the project
            messages: DEPRECATED - List of message dictionaries (for backward compatibility)
            prompt_text: The prompt text (single string)
            prompt_message_type: Type of prompt message (default: "system")
            name: Optional prompt name
            output_schema: Required JSON schema for structured output
            input_schema: Optional JSON schema for input

        Returns:
            prompt_id: ID of the created prompt
        """
        logger.info(f"Creating prompt for project {project_id}")

        # Handle backward compatibility - convert messages array to single prompt_text
        if messages and not prompt_text:
            # Take the first message from the array
            if messages and len(messages) > 0:
                prompt_text = messages[0].get("prompt_text", "")
                prompt_message_type = messages[0].get("prompt_message_type", "system")

        if not prompt_text:
            raise DimRedAPIError("prompt_text is required")

        if not output_schema:
            raise DimRedAPIError("output_schema is required. Please provide a JSON schema for the expected output.")

        data = {
            "project_id": project_id,
            "prompt_text": prompt_text,
            "prompt_message_type": prompt_message_type,
            "output_schema": output_schema
        }

        if input_schema:
            data["input_schema"] = input_schema

        response = self._make_request("POST", "/api/v1/prompts", data=data)

        # The API returns 'id' not 'prompt_id'
        prompt_id = response.get("id") or response.get("prompt_id")

        if not prompt_id:
            logger.error(f"Unexpected response format: {response}")
            raise DimRedAPIError("No prompt_id in response")

        # Cache the prompt details since GET endpoint returns HTML
        self._prompt_cache[prompt_id] = {
            "id": prompt_id,
            "prompt_text": prompt_text,
            "prompt_message_type": prompt_message_type,
            "output_schema": output_schema,
            "input_schema": input_schema,
            "name": name
        }

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

        response = self._make_request("POST", "/api/v1/metrics", data=data)

        # The API returns 'id' not 'metric_id'
        metric_id = response.get("id") or response.get("metric_id")

        if not metric_id:
            logger.error(f"Unexpected response format: {response}")
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

        # Use the new unified workflows API with tune mode
        response = self.run_workflow(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            model_name=model_name,
            provider=provider,
            mode="tune",
            metric_id=metric_id,
            num_iterations=num_iterations,
            include_project_metrics=include_project_metrics
        )

        # The API returns 'id' not 'tuning_session_id'
        tuning_session_id = response.get("id") or response.get("tuning_session_id")
        task_id = response.get("task_id")

        if not tuning_session_id:
            logger.error(f"Unexpected response format: {response}")
            raise DimRedAPIError("Missing session ID in response")

        logger.info(f"✓ Tuning started")
        logger.info(f"  Session ID: {tuning_session_id}")
        logger.info(f"  Task ID: {task_id}")

        return {
            "tuning_session_id": tuning_session_id,
            "task_id": task_id,
            "status": response.get("status", "pending")
        }

    def run_workflow(
        self,
        project_id: str,
        dataset_id: str,
        prompt_id: str,
        model_name: str = "gpt-4o-mini",
        provider: str = "openai",
        mode: str = "inference",
        metric_id: Optional[str] = None,
        num_iterations: int = 1,
        include_project_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Run a workflow with the new v1 API endpoint.

        Args:
            project_id: ID of the project
            dataset_id: ID of the dataset
            prompt_id: ID of the prompt
            model_name: LLM model name
            provider: LLM provider (openai, anthropic, openrouter)
            mode: Workflow mode - "inference", "evaluate", or "tune"
            metric_id: ID of the metric (required for evaluate/tune modes)
            num_iterations: Number of iterations (only for tune mode)
            include_project_metrics: Whether to include project metrics (tune mode)

        Returns:
            Dictionary with workflow response
        """
        logger.info(f"Running workflow in {mode} mode for project {project_id}")
        logger.info(f"  Dataset: {dataset_id}")
        logger.info(f"  Prompt: {prompt_id}")
        if metric_id:
            logger.info(f"  Metric: {metric_id}")
        logger.info(f"  Model: {model_name} ({provider})")

        data = {
            "project_id": project_id,
            "dataset_id": dataset_id,
            "prompt_id": prompt_id,
            "model_name": model_name,
            "provider": provider,
            "mode": mode
        }

        # Add mode-specific parameters
        if mode == "inference":
            # According to API docs, inference mode uses a special dummy metric ID
            # The special metric ID for inference from the docs
            data["metric_id"] = "be8f82bc-52c6-42eb-b151-44abc7d5163d"
        elif mode == "evaluate":
            if not metric_id:
                raise DimRedAPIError("metric_id is required for evaluate mode")
            data["metric_id"] = metric_id
        elif mode == "tune":
            if not metric_id:
                raise DimRedAPIError("metric_id is required for tune mode")
            data["metric_id"] = metric_id
            data["num_iterations"] = num_iterations
            data["includeProjectMetrics"] = include_project_metrics
            logger.info(f"  Iterations: {num_iterations}")

        # Log the request for debugging
        logger.debug(f"Workflow request data: {json.dumps(data, indent=2)}")

        # Make the request to the v1 endpoint using the standard headers
        response = self._make_request("POST", "/api/v1/workflows", data=data)

        logger.info(f"✓ Workflow started in {mode} mode")
        if mode == "tune":
            # The API returns 'id' not 'tuning_session_id'
            session_id = response.get("id") or response.get("tuning_session_id")
            if session_id:
                logger.info(f"  Session ID: {session_id}")

        return response

    def get_tuning_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current status of a tuning session.

        Args:
            session_id: ID of the tuning session

        Returns:
            Dictionary with tuning session details
        """
        # Use the workflows endpoint to get session status
        response = self._make_request(
            "GET",
            f"/api/v1/workflows/{session_id}"
        )
        return response

    def get_prompt(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get a specific prompt by ID.

        Args:
            prompt_id: ID of the prompt

        Returns:
            Dictionary with prompt details

        Note:
            The API endpoint currently returns HTML instead of JSON.
            This method uses a cache to return prompt details when available.
        """
        # First check cache
        if prompt_id in self._prompt_cache:
            return self._prompt_cache[prompt_id]

        # Try to fetch from API
        response = self._make_request(
            "GET",
            f"/api/v1/prompts/{prompt_id}"
        )

        # If we got a valid response, cache and return it
        if response and isinstance(response, dict) and response.get('prompt_text'):
            self._prompt_cache[prompt_id] = response
            return response

        # If we got an empty response (HTML was returned),
        # return a placeholder with just the ID
        if not response:
            return {"id": prompt_id, "note": "Full prompt details not available from API"}

        return response

    def get_best_prompt(self, session_id: str) -> Dict[str, Any]:
        """
        Get the best prompt from a tuning session.

        Args:
            session_id: ID of the tuning session

        Returns:
            Dictionary with best prompt details including:
                - prompt_id: ID of the best prompt
                - messages: List of prompt messages (if available)
                - metrics: Performance metrics
                - iteration: Which iteration this was from
        """
        logger.info(f"Fetching best prompt for session {session_id}")

        # Get the tuning session
        session = self.get_tuning_session(session_id)

        prompt_id = session.get('prompt_id')
        if not prompt_id:
            raise DimRedAPIError("No prompt_id found in tuning session")

        # Build result from session data
        result = {
            "prompt_id": prompt_id,
            "metrics": session.get("metrics"),
            "iteration": session.get("iteration"),
            "is_best": session.get("is_best", False),
            "eval_id": session.get("eval_id")
        }

        # Try to get prompt details if available
        # The session response might already include prompt details
        if session.get("messages"):
            result["messages"] = session.get("messages")
        else:
            # Try to fetch prompt details separately (may not be available)
            try:
                prompt_response = self.get_prompt(prompt_id)
                result["messages"] = prompt_response.get("messages", [])
            except DimRedAPIError as e:
                logger.warning(f"Could not fetch prompt details: {e}")
                logger.info("Prompt details should be available in the workflow/session response")

        logger.info(f"✓ Retrieved best prompt: {prompt_id}")
        return result

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Dictionary with workflow details including status
        """
        response = self._make_request(
            "GET",
            f"/api/v1/workflows/{workflow_id}"
        )
        return response

    def list_workflows(
        self,
        project_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List all workflows for a project.

        Args:
            project_id: ID of the project
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip

        Returns:
            List of workflow dictionaries
        """
        params = {
            "project_id": project_id,
            "limit": limit,
            "offset": offset
        }

        response = self._make_request(
            "GET",
            "/api/v1/workflows",
            params=params
        )

        # Handle paginated response
        if isinstance(response, dict) and "workflows" in response:
            return response["workflows"]
        elif isinstance(response, list):
            return response
        else:
            return []

    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel

        Returns:
            Updated workflow status
        """
        response = self._make_request(
            "POST",
            f"/api/v1/workflows/{workflow_id}/cancel"
        )
        return response

    def wait_for_workflow_completion(
        self,
        workflow_id: str,
        poll_interval: int = 5,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Poll a workflow until it completes or times out.

        Args:
            workflow_id: ID of the workflow
            poll_interval: Seconds between polls (default: 5)
            timeout: Maximum seconds to wait (default: 3600 = 1 hour)

        Returns:
            Final workflow data

        Raises:
            DimRedAPIError: If timeout reached or workflow failed
        """
        logger.info(f"Polling workflow {workflow_id} every {poll_interval}s")
        logger.info(f"Timeout: {timeout}s")

        start_time = time.time()
        last_status = None
        poll_count = 0

        while True:
            elapsed = time.time() - start_time
            poll_count += 1

            if elapsed > timeout:
                raise DimRedAPIError(f"Timeout reached after {timeout}s")

            # Get workflow status
            logger.debug(f"Poll #{poll_count}: Fetching workflow status (elapsed: {elapsed:.1f}s)")
            workflow = self.get_workflow_status(workflow_id)
            status = workflow.get("status", "unknown")

            # Log status changes
            if status != last_status:
                logger.info(f"Status: {status}")
                last_status = status
            else:
                logger.debug(f"Poll #{poll_count}: Status unchanged: {status}")

            # Check if completed
            if status == "completed":
                logger.info("✓ Workflow completed successfully")
                return workflow

            # Check if failed
            if status == "failed":
                error_msg = workflow.get("error_message", "Unknown error")
                raise DimRedAPIError(f"Workflow failed: {error_msg}")

            # Check if cancelled
            if status == "cancelled":
                raise DimRedAPIError("Workflow was cancelled")

            # Wait before next poll
            logger.debug(f"Sleeping for {poll_interval}s before next poll")
            time.sleep(poll_interval)

    def stream_workflow_updates(self, workflow_id: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream real-time updates for a workflow using Server-Sent Events.

        Args:
            workflow_id: ID of the workflow

        Yields:
            Dictionary with update event data

        Note:
            Requires sseclient library: pip install sseclient-py
        """
        if not SSE_AVAILABLE:
            raise DimRedAPIError(
                "SSE support not available. Install sseclient-py: pip install sseclient-py"
            )

        url = f"{self.base_url}/api/v1/workflows/{workflow_id}/updates"
        logger.info(f"Streaming updates for workflow {workflow_id}")

        response = requests.get(url, headers=self.headers, stream=True)
        response.raise_for_status()

        client = sseclient.SSEClient(response)

        for event in client.events():
            try:
                data = json.loads(event.data)
                data["event_type"] = event.event
                yield data

                # Check for terminal states
                if event.event == "error" or data.get("status") in ["completed", "failed", "cancelled"]:
                    break

            except json.JSONDecodeError:
                logger.warning(f"Failed to parse SSE event data: {event.data}")
                continue

    def get_workflow_results(
        self,
        workflow_id: str,
        mode: str
    ) -> Dict[str, Any]:
        """
        Get results for a completed workflow based on its mode.

        Args:
            workflow_id: ID of the workflow
            mode: Workflow mode - "inference", "evaluate", or "tune"

        Returns:
            Dictionary with mode-specific results

        Raises:
            DimRedAPIError: If workflow not completed or mode unknown
        """
        workflow = self.get_workflow_status(workflow_id)

        if workflow.get("status") != "completed":
            raise DimRedAPIError(
                f"Workflow not completed. Status: {workflow.get('status')}"
            )

        if mode == "inference":
            return self._get_inference_results(workflow)
        elif mode == "evaluate":
            return self._get_evaluation_results(workflow)
        elif mode == "tune":
            return self._get_tuning_results(workflow)
        else:
            raise DimRedAPIError(f"Unknown workflow mode: {mode}")

    def _get_inference_results(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get inference mode results.

        Args:
            workflow: Completed workflow object

        Returns:
            Dictionary with inference results and outputs
        """
        project_id = workflow.get("project_id")
        if not project_id:
            return {
                "summary": workflow.get("metrics", {}),
                "runs": [],
                "error": "No project_id in workflow"
            }

        # Find evaluation created by workflow
        try:
            evaluations = self._make_request(
                "GET",
                "/api/v1/evaluations",
                params={"project_id": project_id}
            )

            # Handle different response formats
            if isinstance(evaluations, dict) and "evaluations" in evaluations:
                eval_list = evaluations["evaluations"]
            elif isinstance(evaluations, list):
                eval_list = evaluations
            else:
                eval_list = []

            # Find evaluation for this workflow
            workflow_eval = None
            for eval_obj in eval_list:
                if eval_obj.get("workflow_id") == workflow.get("id"):
                    workflow_eval = eval_obj
                    break

            if not workflow_eval:
                return {
                    "summary": workflow.get("metrics", {}),
                    "runs": [],
                    "note": "No evaluation found for this workflow"
                }

            # Get evaluation runs
            runs = self._make_request(
                "GET",
                f"/api/v1/evaluations/{workflow_eval['id']}/runs"
            )

            # Handle response format
            if isinstance(runs, dict) and "runs" in runs:
                runs = runs["runs"]

            return {
                "summary": workflow.get("metrics", {}),
                "evaluation_id": workflow_eval.get("id"),
                "runs": runs
            }

        except DimRedAPIError as e:
            logger.warning(f"Failed to fetch evaluation runs: {e}")
            return {
                "summary": workflow.get("metrics", {}),
                "runs": [],
                "error": str(e)
            }

    def _get_evaluation_results(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get evaluation mode results with metrics.

        Args:
            workflow: Completed workflow object

        Returns:
            Dictionary with evaluation metrics and detailed runs
        """
        # Get basic inference results
        inference_results = self._get_inference_results(workflow)

        # Add evaluation-specific metrics
        return {
            "metrics": workflow.get("metrics", {}),
            "evaluation_id": inference_results.get("evaluation_id"),
            "detailed_runs": inference_results.get("runs", []),
            "summary": {
                "total_datapoints": len(inference_results.get("runs", [])),
                "workflow_metrics": workflow.get("metrics", {})
            }
        }

    def _get_tuning_results(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get tuning mode results with best prompt.

        Args:
            workflow: Completed workflow object

        Returns:
            Dictionary with tuning iterations and best prompt
        """
        metrics = workflow.get("metrics", {})
        best_prompt_id = metrics.get("best_prompt_id")

        # Try to fetch the best prompt
        best_prompt = None
        if best_prompt_id:
            try:
                # Note: This endpoint might not be available (403)
                best_prompt = self.get_prompt(best_prompt_id)
            except DimRedAPIError as e:
                logger.warning(f"Could not fetch best prompt: {e}")

        return {
            "summary": metrics,
            "best_prompt": best_prompt,
            "best_prompt_id": best_prompt_id,
            "best_iteration": metrics.get("best_iteration"),
            "best_score": metrics.get("best_score"),
            "all_iterations": metrics.get("iterations", []),
            "total_iterations": len(metrics.get("iterations", []))
        }

    def run_workflow_and_wait(
        self,
        project_id: str,
        dataset_id: str,
        prompt_id: str,
        model_name: str = "gpt-4o-mini",
        provider: str = "openai",
        mode: str = "inference",
        metric_id: Optional[str] = None,
        num_iterations: int = 1,
        include_project_metrics: bool = False,
        poll_interval: int = 5,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """
        Run a workflow and wait for completion, returning results.

        This is a convenience method that combines:
        1. Starting the workflow
        2. Waiting for completion
        3. Fetching results

        Args:
            project_id: ID of the project
            dataset_id: ID of the dataset
            prompt_id: ID of the prompt
            model_name: LLM model name
            provider: LLM provider (openai, anthropic, openrouter)
            mode: Workflow mode - "inference", "evaluate", or "tune"
            metric_id: ID of the metric (required for evaluate/tune modes)
            num_iterations: Number of iterations (only for tune mode)
            include_project_metrics: Whether to include project metrics (tune mode)
            poll_interval: Seconds between status polls
            timeout: Maximum seconds to wait for completion

        Returns:
            Dictionary with workflow results
        """
        # Start the workflow
        logger.info(f"Starting {mode} workflow and waiting for completion")
        workflow_response = self.run_workflow(
            project_id=project_id,
            dataset_id=dataset_id,
            prompt_id=prompt_id,
            model_name=model_name,
            provider=provider,
            mode=mode,
            metric_id=metric_id,
            num_iterations=num_iterations,
            include_project_metrics=include_project_metrics
        )

        # Get workflow ID
        workflow_id = workflow_response.get("id")
        if not workflow_id:
            raise DimRedAPIError("No workflow ID in response")

        # Wait for completion
        completed_workflow = self.wait_for_workflow_completion(
            workflow_id=workflow_id,
            poll_interval=poll_interval,
            timeout=timeout
        )

        # Get results
        results = self.get_workflow_results(workflow_id, mode)

        # Add workflow metadata
        results["workflow_id"] = workflow_id
        results["workflow_status"] = completed_workflow.get("status")
        results["completed_at"] = completed_workflow.get("completed_at")
        results["mode"] = mode

        return results

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
