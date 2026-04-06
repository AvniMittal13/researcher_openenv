"""
Two-layer grading system for the Researcher environment.

Layer 1: Structural pre-checks (deterministic, fast, no LLM)
Layer 2: LLM-as-judge with per-task rubric
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GradeResult:
    reward: float
    breakdown: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""


# ------------------------------------------------------------------
# Layer 1 — Structural pre-checks
# ------------------------------------------------------------------

def _count_headers(text: str) -> int:
    return len([ln for ln in text.splitlines() if ln.strip().startswith("#")])


def _count_urls(text: str) -> int:
    return len(re.findall(r"https?://\S+", text))


def _count_images(text: str) -> int:
    return len(re.findall(r"!\[.*?\]\(.*?\)", text))


def _word_count(text: str) -> int:
    return len(text.split())


def check_structural(
    task_number: int,
    output_dir: str,
    code_history: list[dict[str, Any]] | None = None,
    files_created: list[str] | None = None,
) -> tuple[bool, float]:
    """Return (passes_required_checks, bonus_score_0_to_1)."""
    code_history = code_history or []
    files_created = files_created or []

    if task_number == 1:
        path = os.path.join(output_dir, "research_analysis.md")
        if not os.path.isfile(path):
            return False, 0.0
        text = open(path).read()
        if _word_count(text) < 200:
            return False, 0.0
        if _count_headers(text) < 2:
            return False, 0.0
        bonus = min(_count_urls(text), 5) / 5 * 0.5
        bonus += min(_count_headers(text), 6) / 6 * 0.5
        return True, min(bonus, 1.0)

    elif task_number == 2:
        success_runs = [c for c in code_history if c.get("exit_code") == 0]
        if len(success_runs) < 1:
            return False, 0.0
        gen_files = [f for f in files_created if not f.endswith(".md")]
        if len(gen_files) < 1:
            return False, 0.0
        bonus = min(len(success_runs), 5) / 5 * 0.5
        has_plot = any(f.endswith((".png", ".jpg", ".svg")) for f in files_created)
        bonus += 0.5 if has_plot else 0.0
        return True, min(bonus, 1.0)

    elif task_number == 3:
        path = os.path.join(output_dir, "report.md")
        if not os.path.isfile(path):
            return False, 0.0
        text = open(path).read()
        if _word_count(text) < 400:
            return False, 0.0
        if _count_headers(text) < 4:
            return False, 0.0
        bonus = 0.0
        bonus += 0.3 if _count_images(text) >= 1 else 0.0
        bonus += min(_count_urls(text), 5) / 5 * 0.4
        bonus += 0.3 if _word_count(text) >= 600 else 0.0
        return True, min(bonus, 1.0)

    return False, 0.0


# ------------------------------------------------------------------
# Layer 2 — LLM-as-judge
# ------------------------------------------------------------------

_LLM_TIMEOUT = 15
_FALLBACK_SCORE = 0.4
_MAX_CONTENT_CHARS = 3000


def _call_llm_judge(prompt: str) -> dict[str, Any]:
    """Call the LLM judge via OpenAI-compatible API. Returns parsed JSON."""
    api_base = os.environ.get("API_BASE_URL", "")
    api_key = os.environ.get("HF_TOKEN", "not-set")
    model = os.environ.get("MODEL_NAME", "")

    if not api_base or not model:
        logger.warning("API_BASE_URL or MODEL_NAME not set — using fallback score")
        return {"score": _FALLBACK_SCORE, "breakdown": {}, "feedback": "LLM judge unavailable"}

    try:
        from openai import OpenAI

        client = OpenAI(base_url=api_base, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            timeout=_LLM_TIMEOUT,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM judge call failed: %s", exc)
        # Try to extract a score from partial response
        return {"score": _FALLBACK_SCORE, "breakdown": {}, "feedback": f"LLM error: {exc}"}


def _build_grader_content(
    task_number: int,
    output_dir: str,
    code_history: list[dict[str, Any]] | None = None,
    files_created: list[str] | None = None,
) -> str:
    """Build the content string fed to the LLM grader."""
    if task_number == 1:
        path = os.path.join(output_dir, "research_analysis.md")
        return open(path).read()[:_MAX_CONTENT_CHARS]

    elif task_number == 2:
        parts = []
        for i, entry in enumerate(code_history or []):
            parts.append(f"--- Run {i+1} (exit_code={entry.get('exit_code', '?')}) ---")
            parts.append(entry.get("code", "")[:500])
            stdout = entry.get("stdout", "")
            if stdout:
                parts.append(f"stdout: {stdout[:300]}")
        return "\n".join(parts)[:_MAX_CONTENT_CHARS]

    elif task_number == 3:
        path = os.path.join(output_dir, "report.md")
        return open(path).read()[:_MAX_CONTENT_CHARS]

    return ""


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def grade_task(
    task_number: int,
    research_question: str,
    output_dir: str,
    code_history: list[dict[str, Any]] | None = None,
    files_created: list[str] | None = None,
) -> GradeResult:
    """Grade the output of a task. Returns reward in [0.0, 1.0]."""
    try:
        from researcher1.tasks.task_registry import GRADER_PROMPTS
    except ModuleNotFoundError:
        from tasks.task_registry import GRADER_PROMPTS

    # Layer 1
    ok, structural_bonus = check_structural(
        task_number, output_dir, code_history, files_created
    )
    if not ok:
        return GradeResult(
            reward=0.0,
            breakdown={"reason": "structural checks failed"},
            feedback="Required output missing or too short",
        )

    # Layer 2
    content = _build_grader_content(task_number, output_dir, code_history, files_created)
    prompt_template = GRADER_PROMPTS[task_number]
    prompt = prompt_template.format(
        research_question=research_question,
        content=content,
        files=", ".join(files_created or []),
    )

    llm_result = _call_llm_judge(prompt)
    llm_score = float(llm_result.get("score", _FALLBACK_SCORE))
    llm_score = max(0.0, min(1.0, llm_score))

    final = llm_score * 0.85 + structural_bonus * 0.15
    return GradeResult(
        reward=min(final, 1.0),
        breakdown=llm_result.get("breakdown", {}),
        feedback=llm_result.get("feedback", ""),
    )
