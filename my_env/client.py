# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dark Store Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DarkStoreAction, DarkStoreObservation


class DarkStoreClient(
    EnvClient[DarkStoreAction, DarkStoreObservation, State]
):
    """
    Client for the Dark Store Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with DarkStoreClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.tick)
        ...
        ...     result = client.step(DarkStoreAction(action="wait"))
        ...     print(result.observation.tick)

    Example with Docker:
        >>> client = DarkStoreClient.from_docker_image("dark_store-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(DarkStoreAction(action="wait"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: DarkStoreAction) -> Dict:
        """
        Convert DarkStoreAction to JSON payload for step message.

        Args:
            action: DarkStoreAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict = {"action": action.action}
        if action.row is not None:
            payload["row"] = action.row
        if action.col is not None:
            payload["col"] = action.col
        if action.item_name is not None:
            payload["item_name"] = action.item_name
        if action.order_id is not None:
            payload["order_id"] = action.order_id
        if action.rider_id is not None:
            payload["rider_id"] = action.rider_id
        if action.order_a is not None:
            payload["order_a"] = action.order_a
        if action.order_b is not None:
            payload["order_b"] = action.order_b
        if action.quantity is not None:
            payload["quantity"] = action.quantity
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DarkStoreObservation]:
        """
        Parse server response into StepResult[DarkStoreObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with DarkStoreObservation
        """
        obs_data = payload.get("observation", {})
        observation = DarkStoreObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id, step_count, and task_name
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name"),
        )
