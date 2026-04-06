"""
Researcher environment — implements the 3-task research pipeline on the
OpenEnv Environment base class.

Task 1: Topic summary, analysis & candidate comparison  → research_analysis.md
Task 2: Experimentation & coding                        → code outputs + plots
Task 3: Complete research report                        → report.md

Each task is a separate episode (reset → step* → finalize).
Output directory persists across tasks for the same topic so that
Task 2 can read Task 1's output and Task 3 can read both.

NOTE: The OpenEnv HTTP server creates a NEW env instance per request.
All state is therefore persisted to disk (JSON sidecar file) so it
survives across HTTP calls.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from researcher1.env.models import ResearchAction, ResearchObservation, ResearchState
    from researcher1.graders.research_grader import grade_task
    from researcher1.tasks.task_registry import get_topic
    from researcher1.tools import code_executor, scraper, web_search
except ModuleNotFoundError:
    from env.models import ResearchAction, ResearchObservation, ResearchState
    from graders.research_grader import grade_task
    from tasks.task_registry import get_topic
    from tools import code_executor, scraper, web_search

_BASE_OUTPUT_DIR = os.environ.get("RESEARCHER_OUTPUT_DIR", "/tmp/researcher")
_STATE_FILE = "._env_state.json"

# Partial reward caps per action type
_PARTIAL_CAPS = {
    "web_search": (0.05, 3),     # +0.05 each, max 3
    "scrape_url": (0.05, 3),     # +0.05 each, max 3
    "execute_code": (0.03, 3),   # +0.03 each (if exit_code==0), max 3
    "take_notes": (0.05, 2),     # +0.05 each, max 2
    "save_file": (0.02, 2),      # +0.02 each, max 2
}


def _active_output_dir() -> str:
    """Return the output dir for the currently active session."""
    marker = os.path.join(_BASE_OUTPUT_DIR, "._active_topic")
    if os.path.isfile(marker):
        return os.path.join(_BASE_OUTPUT_DIR, open(marker).read().strip())
    return ""


def _save_session(output_dir: str, state: ResearchState,
                  cumulative_reward: float, action_counts: dict) -> None:
    """Persist session state to a JSON sidecar file."""
    path = os.path.join(output_dir, _STATE_FILE)
    data = {
        "state": state.model_dump(),
        "cumulative_reward": cumulative_reward,
        "action_counts": action_counts,
    }
    with open(path, "w") as f:
        json.dump(data, f)


def _load_session(output_dir: str) -> tuple[ResearchState, float, dict] | None:
    """Load session state from disk. Returns None if no session exists."""
    path = os.path.join(output_dir, _STATE_FILE)
    if not os.path.isfile(path):
        return None
    try:
        data = json.loads(open(path).read())
        state = ResearchState(**data["state"])
        return state, data["cumulative_reward"], data["action_counts"]
    except Exception:
        return None


class ResearcherEnv(Environment[ResearchAction, ResearchObservation, ResearchState]):
    """
    Researcher agent environment.

    reset() → initial observation with the research question
    step()  → execute one tool action, return observation
    state   → full internal state for inspection
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = ResearchState(episode_id="")
        self._output_dir = ""
        self._cumulative_reward = 0.0
        self._action_counts: dict[str, int] = {}
        # Try to restore from active session on disk
        self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        """Load state from disk if an active session exists."""
        output_dir = _active_output_dir()
        if not output_dir:
            return
        loaded = _load_session(output_dir)
        if loaded is None:
            return
        state, cumulative_reward, action_counts = loaded
        self._state = state
        self._output_dir = output_dir
        self._cumulative_reward = cumulative_reward
        self._action_counts = action_counts

    def _persist(self) -> None:
        """Save current state to disk."""
        if self._output_dir:
            _save_session(self._output_dir, self._state,
                          self._cumulative_reward, self._action_counts)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        *,
        task_number: int = 1,
        topic_id: str = "topic_01",
        max_steps: int = 20,
        **kwargs: Any,
    ) -> ResearchObservation:
        topic = get_topic(topic_id)

        # Output dir keyed by topic — persists across tasks
        self._output_dir = os.path.join(_BASE_OUTPUT_DIR, topic_id)
        os.makedirs(self._output_dir, exist_ok=True)

        eid = episode_id or uuid.uuid4().hex[:12]
        self._state = ResearchState(
            episode_id=eid,
            step_count=0,
            task_number=task_number,
            topic_id=topic_id,
            research_question=topic.research_area,
            max_steps=max_steps,
        )
        self._cumulative_reward = 0.0
        self._action_counts = {}

        # Mark this topic as the active session
        os.makedirs(_BASE_OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(_BASE_OUTPUT_DIR, "._active_topic"), "w") as f:
            f.write(topic_id)

        # Build initial context
        context = f"Research area: {topic.research_area}"
        if task_number >= 2:
            analysis_path = os.path.join(self._output_dir, "research_analysis.md")
            if os.path.isfile(analysis_path):
                content = open(analysis_path).read()
                context += f"\n\n--- research_analysis.md (from Task 1) ---\n{content[:4000]}"
        if task_number == 3:
            # Include code history summary and file listing
            files = sorted(os.listdir(self._output_dir))
            context += f"\n\n--- Files from prior tasks ---\n{', '.join(files)}"

        self._persist()

        return ResearchObservation(
            action_type="reset",
            success=True,
            result=context,
            files_created=self._list_files(),
            steps_remaining=max_steps,
            cumulative_reward=0.0,
            reward=0.0,
            done=False,
        )

    # ------------------------------------------------------------------
    # step — restores state from disk on each call (new instance per HTTP req)
    # ------------------------------------------------------------------
    def step(
        self,
        action: ResearchAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ResearchObservation:
        # Restore state (HTTP server creates new instance per request)
        self._restore_from_disk()
        s = self._state
        s.step_count += 1
        at = action.action_type

        # Dispatch to tool
        try:
            if at == "web_search":
                obs = self._do_web_search(action)
            elif at == "scrape_url":
                obs = self._do_scrape(action)
            elif at == "execute_code":
                obs = self._do_execute_code(action)
            elif at == "take_notes":
                obs = self._do_take_notes(action)
            elif at == "save_file":
                obs = self._do_save_file(action)
            elif at == "finalize":
                obs = self._do_finalize()
            else:
                obs = self._make_obs(
                    action_type=at, success=False,
                    result="", error=f"Unknown action_type: {at}",
                )
        except Exception as exc:
            obs = self._make_obs(
                action_type=at, success=False,
                result="", error=f"Tool error: {exc}",
            )

        # Auto-finalize if we hit max steps and haven't finalized
        if s.step_count >= s.max_steps and not s.is_done:
            obs = self._do_finalize()

        self._persist()
        return self._apply_transform(obs)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> ResearchState:
        return self._state

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------
    def _do_web_search(self, action: ResearchAction) -> ResearchObservation:
        query = action.query or ""
        if not query:
            return self._make_obs("web_search", False, "", error="query is required")

        results = web_search.search(query)
        self._state.search_results.append({"query": query, "results": results})
        self._add_partial("web_search")

        return self._make_obs(
            "web_search", True,
            result=json.dumps(results, indent=2)[:3000],
        )

    def _do_scrape(self, action: ResearchAction) -> ResearchObservation:
        url = action.url or ""
        if not url:
            return self._make_obs("scrape_url", False, "", error="url is required")

        data = scraper.scrape(url)
        preview = data.get("content", "")[:200]
        self._state.scraped_pages.append({
            "url": url, "type": data.get("type", "unknown"),
            "content_preview": preview,
        })
        self._add_partial("scrape_url")

        return self._make_obs(
            "scrape_url", True,
            result=data.get("content", "")[:2000],
        )

    def _do_execute_code(self, action: ResearchAction) -> ResearchObservation:
        code = action.code or ""
        if not code:
            return self._make_obs("execute_code", False, "", error="code is required")

        result = code_executor.execute(code, self._output_dir)
        self._state.code_history.append({
            "code": code[:1000],
            "stdout": result["stdout"][:500],
            "stderr": result["stderr"][:200],
            "exit_code": result["exit_code"],
        })
        self._state.files_created = self._list_files()
        if result["exit_code"] == 0:
            self._add_partial("execute_code")

        return self._make_obs(
            "execute_code", result["exit_code"] == 0,
            result=json.dumps({
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"],
                "files": result["files"],
            }),
            error=result["stderr"] if result["exit_code"] != 0 else None,
        )

    def _do_take_notes(self, action: ResearchAction) -> ResearchObservation:
        notes = action.notes or ""
        if not notes:
            return self._make_obs("take_notes", False, "", error="notes is required")

        self._state.notes += notes + "\n\n"
        self._add_partial("take_notes")

        return self._make_obs(
            "take_notes", True,
            result=f"Notes appended. Total notes length: {len(self._state.notes)} chars",
        )

    def _do_save_file(self, action: ResearchAction) -> ResearchObservation:
        filename = action.filename or ""
        content = action.content or ""
        if not filename:
            return self._make_obs("save_file", False, "", error="filename is required")
        if not content:
            return self._make_obs("save_file", False, "", error="content is required")

        # Sanitize filename — no path traversal
        safe_name = os.path.basename(filename)
        path = os.path.join(self._output_dir, safe_name)
        with open(path, "w") as f:
            f.write(content[:50000])  # cap at 50KB

        self._state.files_created = self._list_files()
        self._add_partial("save_file")

        return self._make_obs(
            "save_file", True,
            result=f"Saved {safe_name} ({len(content)} chars)",
        )

    def _do_finalize(self) -> ResearchObservation:
        s = self._state
        s.is_done = True

        # Auto-save notes as the expected output file if agent forgot to save_file
        if s.task_number == 1 and s.notes:
            path = os.path.join(self._output_dir, "research_analysis.md")
            if not os.path.isfile(path):
                with open(path, "w") as f:
                    f.write(s.notes)
                s.files_created = self._list_files()
        elif s.task_number == 3 and s.notes:
            path = os.path.join(self._output_dir, "report.md")
            if not os.path.isfile(path):
                with open(path, "w") as f:
                    f.write(s.notes)
                s.files_created = self._list_files()

        grade = grade_task(
            task_number=s.task_number,
            research_question=s.research_question,
            output_dir=self._output_dir,
            code_history=s.code_history,
            files_created=s.files_created,
        )
        s.final_reward = grade.reward
        final_reward = self._cumulative_reward + grade.reward

        return ResearchObservation(
            action_type="finalize",
            success=True,
            result=json.dumps({
                "final_reward": final_reward,
                "grader_reward": grade.reward,
                "partial_reward": self._cumulative_reward,
                "breakdown": grade.breakdown,
                "feedback": grade.feedback,
            }),
            files_created=self._list_files(),
            steps_remaining=0,
            cumulative_reward=final_reward,
            reward=final_reward,
            done=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _add_partial(self, action_type: str) -> None:
        if action_type not in _PARTIAL_CAPS:
            return
        per_action, cap = _PARTIAL_CAPS[action_type]
        count = self._action_counts.get(action_type, 0)
        if count < cap:
            self._cumulative_reward += per_action
            self._action_counts[action_type] = count + 1

    def _list_files(self) -> list[str]:
        if os.path.isdir(self._output_dir):
            return sorted(
                f for f in os.listdir(self._output_dir) if not f.startswith("._")
            )
        return []

    def _make_obs(
        self,
        action_type: str,
        success: bool,
        result: str,
        error: str | None = None,
    ) -> ResearchObservation:
        s = self._state
        return ResearchObservation(
            action_type=action_type,
            success=success,
            result=result,
            error=error,
            files_created=self._list_files(),
            steps_remaining=max(s.max_steps - s.step_count, 0),
            cumulative_reward=self._cumulative_reward,
            reward=self._cumulative_reward,
            done=s.is_done,
        )
