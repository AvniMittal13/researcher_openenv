"""
Pydantic models for the Researcher environment.

Follows the OpenEnv 3-component pattern: models → server → client.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ResearchAction(Action):
    """A single action the research agent can take."""

    action_type: Literal[
        "web_search", "scrape_url", "execute_code",
        "take_notes", "save_file", "finalize",
    ] = Field(..., description="Which tool to invoke")

    query: Optional[str] = Field(
        default=None, description="Search query (for web_search)"
    )
    url: Optional[str] = Field(
        default=None, description="URL to scrape (for scrape_url)"
    )
    code: Optional[str] = Field(
        default=None, description="Python code to execute (for execute_code)"
    )
    notes: Optional[str] = Field(
        default=None, description="Notes to append (for take_notes)"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Filename to save (for save_file), e.g. 'research_analysis.md'",
    )
    content: Optional[str] = Field(
        default=None, description="File content to write (for save_file)"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ResearchObservation(Observation):
    """What the agent sees after each action."""

    action_type: str = Field(
        default="reset", description="Which action was just taken"
    )
    success: bool = Field(default=True, description="Did the action succeed?")
    result: str = Field(
        default="", description="Action-specific output text"
    )
    files_created: List[str] = Field(
        default_factory=list,
        description="Files currently in the output directory",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if action failed"
    )
    steps_remaining: int = Field(
        default=20, description="max_steps - steps_taken"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Running partial reward"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(State):
    """Full internal state of one Researcher episode."""

    task_number: int = Field(
        default=1, description="Which task (1=analysis, 2=experiment, 3=report)"
    )
    topic_id: str = Field(default="", description="Topic identifier")
    research_question: str = Field(
        default="", description="The research area / question"
    )
    search_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{query, results: [{title, url, snippet}]}]",
    )
    scraped_pages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{url, type, content_preview}]",
    )
    notes: str = Field(default="", description="Accumulated notes")
    files_created: List[str] = Field(
        default_factory=list, description="Files in output dir"
    )
    code_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="[{code, stdout, stderr, exit_code}]",
    )
    max_steps: int = Field(default=20)
    is_done: bool = Field(default=False)
    final_reward: Optional[float] = Field(default=None)
