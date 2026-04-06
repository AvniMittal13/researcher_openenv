"""Researcher Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from researcher1.env.models import ResearchAction, ResearchObservation, ResearchState
except ModuleNotFoundError:
    from env.models import ResearchAction, ResearchObservation, ResearchState


class ResearcherEnvClient(
    EnvClient[ResearchAction, ResearchObservation, ResearchState]
):
    """
    Client for the Researcher Environment.

    Example:
        >>> with ResearcherEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_number=1, topic_id="topic_01")
        ...     result = client.step(ResearchAction(action_type="web_search", query="RAG"))
    """

    def _step_payload(self, action: ResearchAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ResearchObservation]:
        obs_data = payload.get("observation", {})
        observation = ResearchObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ResearchState:
        return ResearchState(**payload)
