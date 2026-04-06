"""Task registry — loads topics and provides per-task configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Topic:
    topic_id: str
    research_area: str
    expected_keywords: list[str] = field(default_factory=list)


def load_topics() -> list[Topic]:
    path = _DATA_DIR / "topics.json"
    raw = json.loads(path.read_text())
    return [Topic(**t) for t in raw]


def get_topic(topic_id: str) -> Topic:
    for t in load_topics():
        if t.topic_id == topic_id:
            return t
    raise ValueError(f"Unknown topic_id: {topic_id}")


# ------------------------------------------------------------------
# Per-task LLM grader prompts
# ------------------------------------------------------------------

GRADER_PROMPTS: dict[int, str] = {
    1: """\
You are grading a research analysis document. The research area was: "{research_question}"

The document should contain:
1. A clear background summary with cited sources (0-0.2)
2. At least 3 candidate topics identified with pros/cons (0-0.25)
3. A structured comparison (table or detailed prose) (0-0.2)
4. A justified selection of one topic for further research (0-0.15)
5. Overall quality: clarity, depth, accuracy (0-0.2)

Document:
{content}

Respond with ONLY valid JSON (no markdown fences):
{{"score": 0.0, "breakdown": {{"background": 0.0, "candidates": 0.0, "comparison": 0.0, "selection": 0.0, "quality": 0.0}}, "feedback": "one sentence"}}""",

    2: """\
You are grading a coding and experimentation session for a research project.
Research topic: "{research_question}"

Evaluate based on:
1. Code quality: Does the code run successfully? Is it relevant to the topic? (0-0.25)
2. Experimental rigor: Are the experiments meaningful, not just trivial? (0-0.25)
3. Results produced: Are there concrete outputs (numbers, tables, data)? (0-0.2)
4. Visualizations: Is at least one chart/plot generated and relevant? (0-0.15)
5. Reproducibility: Could someone re-run this and get similar results? (0-0.15)

Code history:
{content}

Files created: {files}

Respond with ONLY valid JSON (no markdown fences):
{{"score": 0.0, "breakdown": {{"code_quality": 0.0, "rigor": 0.0, "results": 0.0, "visualizations": 0.0, "reproducibility": 0.0}}, "feedback": "one sentence"}}""",

    3: """\
You are grading a complete research report. The topic is: "{research_question}"

Evaluate based on:
1. Structure: Does it have abstract, intro, lit review, methodology, results, discussion, conclusion, references? (0-0.15)
2. Literature coverage: Are sources properly cited and summarized? (0-0.15)
3. Methodology clarity: Is the experimental approach clearly described? (0-0.15)
4. Results presentation: Are experimental results included with data/charts? (0-0.15)
5. Analysis depth: Is there meaningful interpretation, not just description? (0-0.15)
6. Writing quality: Coherent, well-organized, no obvious errors? (0-0.1)
7. Completeness: Does the report feel like a finished product, not a draft? (0-0.15)

Report:
{content}

Respond with ONLY valid JSON (no markdown fences):
{{"score": 0.0, "breakdown": {{"structure": 0.0, "literature": 0.0, "methodology": 0.0, "results": 0.0, "analysis": 0.0, "writing": 0.0, "completeness": 0.0}}, "feedback": "one sentence"}}""",
}
