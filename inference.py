"""
Inference Script for ResearcherEnv
===================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory
- Uses OpenAI Client for all LLM calls
- Emits structured stdout logs: [START], [STEP], [END]

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.groq.com/openai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "llama-3.1-8b-instant"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "researcher_env"
MAX_STEPS = 20

# ------------------------------------------------------------------
# Task-specific system prompts
# ------------------------------------------------------------------

TASK_PROMPTS = {
    1: """\
You are a research agent. Your task is to research a topic area, identify candidate sub-topics, compare them, and produce a research analysis.

Tools available (respond with ONE JSON action per step):
- {"action_type": "web_search", "query": "..."} — Search the web
- {"action_type": "scrape_url", "url": "..."} — Read a webpage or PDF
- {"action_type": "take_notes", "notes": "..."} — Accumulate research notes
- {"action_type": "save_file", "filename": "research_analysis.md", "content": "..."} — Save a file
- {"action_type": "finalize"} — Submit for grading (call when done)

Workflow:
1. Do 2-3 web searches about the research area
2. Scrape 2-3 of the most relevant URLs from search results
3. Take notes summarizing key findings from your research
4. YOU MUST call save_file to save research_analysis.md BEFORE calling finalize
5. Call finalize ONLY after saving the file

CRITICAL: You MUST use save_file with filename "research_analysis.md" before finalize.
The file should have sections: Background, Sources, Candidate Topics, Comparison, Selected Topic.

IMPORTANT: Respond with ONLY a single JSON object per step. No other text.""",

    2: """\
You are a research agent. Your task is to conduct coding experiments on a selected research topic.

The file research_analysis.md contains your prior analysis with a selected topic. Use it as context.

Tools available (respond with ONE JSON action per step):
- {"action_type": "web_search", "query": "..."} — Search for code examples, datasets, APIs
- {"action_type": "scrape_url", "url": "..."} — Read documentation or tutorials
- {"action_type": "execute_code", "code": "..."} — Run Python code (OUTPUT_DIR is pre-set; save plots/data there)
- {"action_type": "take_notes", "notes": "..."} — Record observations
- {"action_type": "finalize"} — Submit for grading (call when done)

Workflow:
1. Read the research_analysis.md context provided in the initial observation
2. Search for 1-2 code examples or libraries relevant to the selected topic
3. Write and execute Python code that experiments with the topic (use execute_code). Save plots to OUTPUT_DIR using: plt.savefig(os.path.join(OUTPUT_DIR, 'chart.png'))
4. Take notes on your experimental findings
5. Call finalize

CRITICAL: You MUST use execute_code to run at least one experiment and generate at least one plot (.png). 
Use matplotlib with: import matplotlib; matplotlib.use('Agg')

IMPORTANT: Respond with ONLY a single JSON object per step. No other text.""",

    3: """\
You are a research agent. Your task is to write a complete, polished research report.

You have access to research_analysis.md (literature review) and experimental results from prior tasks.

Tools available (respond with ONE JSON action per step):
- {"action_type": "web_search", "query": "..."} — Search for additional context if needed
- {"action_type": "execute_code", "code": "..."} — Run code for additional analysis/plots if needed
- {"action_type": "save_file", "filename": "report.md", "content": "..."} — Save the final report
- {"action_type": "finalize"} — Submit for grading (call when done)

Workflow:
1. Review the research_analysis.md and file listing in the initial observation
2. Write a comprehensive report.md (500+ words) with sections: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Conclusion, References
3. Use save_file to save report.md BEFORE calling finalize
4. Call finalize

CRITICAL: You MUST call save_file with filename "report.md" before finalize.
Reference any generated plots like: ![chart](chart.png)

IMPORTANT: Respond with ONLY a single JSON object per step. No other text.""",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def parse_action(text: str) -> dict[str, Any]:
    """Extract a JSON action from the LLM response text."""
    # Try direct parse
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: finalize
    return {"action_type": "finalize"}


def _flatten_response(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten the OpenEnv HTTP response into a simple observation dict.

    The server returns: {"observation": {...fields...}, "reward": X, "done": Y}
    We merge into: {...fields..., "reward": X, "done": Y}
    """
    obs = data.get("observation", {})
    obs["reward"] = data.get("reward", 0.0)
    obs["done"] = data.get("done", False)
    return obs


def env_reset(env_url: str, task_number: int, topic_id: str) -> dict[str, Any]:
    resp = requests.post(
        f"{env_url}/reset",
        json={"task_number": task_number, "topic_id": topic_id},
    )
    resp.raise_for_status()
    return _flatten_response(resp.json())


def env_step(env_url: str, action: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(f"{env_url}/step", json={"action": action})
    resp.raise_for_status()
    return _flatten_response(resp.json())


# ------------------------------------------------------------------
# Logging helpers (exact hackathon format)
# ------------------------------------------------------------------

TASK_NAMES = {1: "topic_analysis", 2: "experimentation", 3: "research_report"}


def log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ------------------------------------------------------------------
# Episode runner
# ------------------------------------------------------------------

def run_episode(
    env_url: str,
    task_number: int,
    topic_id: str,
    client: OpenAI,
) -> float:
    """Run one task episode. Returns the final score in [0, 1]."""
    task_name = TASK_NAMES.get(task_number, f"task_{task_number}")
    log_start(task_name=task_name, model=MODEL_NAME)

    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = env_reset(env_url, task_number, topic_id)
        system_prompt = TASK_PROMPTS[task_number]
        messages = [{"role": "system", "content": system_prompt}]

        while not obs.get("done", False) and steps_taken < MAX_STEPS:
            # Build user message from observation — cap to avoid token limits
            result_text = obs.get("result", "")[:2000]
            user_msg = f"Observation:\n{result_text}"
            if obs.get("error"):
                user_msg += f"\n\nError: {obs['error']}"
            user_msg += f"\n\nSteps remaining: {obs.get('steps_remaining', '?')}"
            user_msg += f"\nFiles in output: {obs.get('files_created', [])}"

            # Keep only system + last 4 exchanges to stay under token limits
            if len(messages) > 9:
                messages = messages[:1] + messages[-8:]

            messages.append({"role": "user", "content": user_msg})

            # Get LLM response with retry on rate limits
            llm_text = ""
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4096,
                    )
                    llm_text = response.choices[0].message.content or ""
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower() or "too many" in str(e).lower():
                        wait = 2 ** attempt
                        time.sleep(wait)
                    else:
                        llm_text = '{"action_type": "finalize"}'
                        break
            messages.append({"role": "assistant", "content": llm_text})

            # Parse and execute action
            action = parse_action(llm_text)
            action_type = action.get("action_type", "finalize")

            obs = env_step(env_url, action)
            steps_taken += 1

            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)
            error = obs.get("error", None)
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=action_type,
                reward=reward,
                done=done,
                error=error,
            )

        # Compute score in [0, 1]
        score = obs.get("cumulative_reward", 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score > 0.0

    except Exception as exc:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(exc))
        rewards.append(0.0)
        steps_taken += 1

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Researcher agent inference")
    parser.add_argument("--topic", default="topic_01", help="Topic ID")
    parser.add_argument("--env-url", default=ENV_URL, help="Environment server URL")
    parser.add_argument("--tasks", default="1,2,3", help="Comma-separated task numbers to run")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = [int(t) for t in args.tasks.split(",")]

    total_score = 0.0
    for task_num in tasks:
        score = run_episode(args.env_url, task_num, args.topic, client)
        total_score += score

    avg_score = total_score / len(tasks) if tasks else 0.0
    print(f"\nPipeline complete: avg_score={avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()
