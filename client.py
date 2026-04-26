"""
AnomalyCraft Survival — OpenEnv EnvClient subclass.

Provides a WebSocket-based client for interacting with the SurvivalEnv server.
Supports both async (default) and sync (via .sync()) usage.

Usage (async):
    async with SurvivalClient(base_url="ws://localhost:8000") as env:
        result = await env.reset(task_id=101)
        while not result.done:
            action = SurvivalAction(agent_id="agent_1", action_type="gather")
            result = await env.step(action)

Usage (sync):
    client = SurvivalClient(base_url="ws://localhost:8000").sync()
    with client:
        result = client.reset(task_id=101)
        result = client.step(SurvivalAction(agent_id="agent_1", action_type="gather"))
"""

from __future__ import annotations

from typing import Any, Dict

from openenv_core.http_env_client import HTTPEnvClient as EnvClient
from openenv_core.client_types import StepResult

from models import SurvivalAction, SurvivalObservation, SurvivalWorldState


class SurvivalClient(EnvClient[SurvivalAction, SurvivalObservation, SurvivalWorldState]):
    """
    WebSocket client for AnomalyCraft Survival environment.
    Inherits from openenv.core.EnvClient — async by default.
    Use .sync() for synchronous access.
    """

    def _step_payload(self, action: SurvivalAction) -> Dict[str, Any]:
        """Convert SurvivalAction to the JSON payload the server expects."""
        if hasattr(action, "model_dump"):
            return action.model_dump(exclude_none=True)
        return dict(action)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SurvivalObservation]:
        """Parse server response into StepResult[SurvivalObservation]."""
        try:
            obs = SurvivalObservation(**payload)
        except Exception:
            # Fallback: wrap raw payload
            obs = payload  # type: ignore[assignment]
        return StepResult(
            observation=obs,
            reward=payload.get("reward_update", payload.get("reward")),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SurvivalWorldState:
        """Parse server state response into SurvivalWorldState."""
        try:
            return SurvivalWorldState(**payload)
        except Exception:
            return payload  # type: ignore[return-value]
