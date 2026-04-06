"""
Smoke tests for the Researcher environment.

Runs without network or LLM — tests environment mechanics,
tool dispatch, grader structural checks, and episode flow.
"""

import json
import os
import shutil
import sys
import tempfile

# Patch output dir to a temp location before importing env
_TEMP_DIR = tempfile.mkdtemp(prefix="researcher_test_")
os.environ["RESEARCHER_OUTPUT_DIR"] = _TEMP_DIR

try:
    from researcher1.env.models import ResearchAction
    from researcher1.env.researcher_env import ResearcherEnv
    from researcher1.graders.research_grader import check_structural
    from researcher1.tasks.task_registry import get_topic, load_topics
except ModuleNotFoundError:
    from env.models import ResearchAction
    from env.researcher_env import ResearcherEnv
    from graders.research_grader import check_structural
    from tasks.task_registry import get_topic, load_topics


def cleanup():
    shutil.rmtree(_TEMP_DIR, ignore_errors=True)


# ===================================================================
print("=== Topic Registry Tests ===")

topics = load_topics()
assert len(topics) == 5, f"Expected 5 topics, got {len(topics)}"
t = get_topic("topic_01")
assert "RAG" in t.research_area
print(f"  load_topics OK: {len(topics)} topics")
print(f"  get_topic OK: {t.topic_id} = {t.research_area[:50]}...")

# ===================================================================
print("\n=== Environment Reset Tests ===")

env = ResearcherEnv()

# Task 1 reset
obs = env.reset(task_number=1, topic_id="topic_01")
assert not obs.done
assert obs.reward == 0.0
assert "RAG" in obs.result
assert obs.steps_remaining == 20
print(f"  Task 1 reset OK: result contains topic, done={obs.done}")

# Task 2 reset without prior Task 1 output
obs2 = env.reset(task_number=2, topic_id="topic_01")
assert not obs2.done
assert "RAG" in obs2.result
print(f"  Task 2 reset OK (no prior output): done={obs2.done}")

# ===================================================================
print("\n=== Step: take_notes ===")

env.reset(task_number=1, topic_id="topic_02")
obs = env.step(ResearchAction(action_type="take_notes", notes="LoRA is great"))
assert obs.success
assert obs.action_type == "take_notes"
assert obs.cumulative_reward > 0
assert env.state.notes.startswith("LoRA is great")
print(f"  take_notes OK: reward={obs.cumulative_reward:.2f}, notes_len={len(env.state.notes)}")

# ===================================================================
print("\n=== Step: save_file ===")

obs = env.step(ResearchAction(
    action_type="save_file",
    filename="research_analysis.md",
    content="# Analysis\n\n## Background\nSome content here about LoRA.\n\n## Candidates\n...",
))
assert obs.success
assert "research_analysis.md" in obs.files_created
print(f"  save_file OK: files={obs.files_created}")

# ===================================================================
print("\n=== Step: execute_code ===")

obs = env.step(ResearchAction(
    action_type="execute_code",
    code="print('hello from test')\nwith open(os.path.join(OUTPUT_DIR, 'test.txt'), 'w') as f:\n    f.write('data')",
))
assert obs.success
result = json.loads(obs.result)
assert result["exit_code"] == 0
assert "hello from test" in result["stdout"]
assert "test.txt" in result["files"]
print(f"  execute_code OK: exit={result['exit_code']}, files={result['files']}")

# ===================================================================
print("\n=== Step: bad action ===")

obs = env.step(ResearchAction(action_type="web_search"))  # missing query
assert not obs.success
assert obs.error is not None
print(f"  bad action OK: error={obs.error}")

# ===================================================================
print("\n=== Step: save_file path traversal protection ===")

obs = env.step(ResearchAction(
    action_type="save_file",
    filename="../../../etc/passwd",
    content="malicious",
))
assert obs.success  # It succeeds but sanitizes the filename
assert "passwd" in obs.result  # saved as just "passwd"
# Verify it didn't actually write outside the output dir
assert not os.path.exists("/etc/passwd_test")
print(f"  path traversal protection OK: {obs.result}")

# ===================================================================
print("\n=== Structural Grader Tests ===")

# Task 1: create a proper research_analysis.md
topic_dir = os.path.join(_TEMP_DIR, "topic_test_grader")
os.makedirs(topic_dir, exist_ok=True)

# Too short → fail
with open(os.path.join(topic_dir, "research_analysis.md"), "w") as f:
    f.write("Short")
ok, bonus = check_structural(1, topic_dir)
assert not ok
print("  Task 1 structural (too short): FAIL as expected")

# Good enough → pass
with open(os.path.join(topic_dir, "research_analysis.md"), "w") as f:
    f.write(
        "# Research Analysis\n\n## Background\n"
        + "Some detailed content about the topic. " * 50
        + "\n\n## Candidates\n"
        + "More content here. " * 30
        + "\n\nhttps://example.com\nhttps://arxiv.org/abs/1234\nhttps://github.com/test"
    )
ok, bonus = check_structural(1, topic_dir)
assert ok
print(f"  Task 1 structural (good): PASS, bonus={bonus:.2f}")

# Task 2: code history checks
ok2, bonus2 = check_structural(
    2, topic_dir,
    code_history=[{"exit_code": 0, "code": "print(1)", "stdout": "1", "stderr": ""}],
    files_created=["chart.png"],
)
assert ok2
print(f"  Task 2 structural: PASS, bonus={bonus2:.2f}")

# Task 2: no successful runs → fail
ok2f, _ = check_structural(
    2, topic_dir,
    code_history=[{"exit_code": 1, "code": "bad", "stdout": "", "stderr": "error"}],
    files_created=[],
)
assert not ok2f
print("  Task 2 structural (no success): FAIL as expected")

# Task 3: create report.md
with open(os.path.join(topic_dir, "report.md"), "w") as f:
    f.write(
        "# Report\n\n## Abstract\nThis is a detailed summary of the research. " * 5
        + "\n\n## Introduction\n"
        + "Background content about the research topic area is described here in detail. " * 40
        + "\n\n## Methodology\nThe experimental approach involved several key steps. " * 20
        + "\n\n## Results\n"
        + "The experimental findings demonstrate several important insights into the topic. " * 30
        + "\n\n## Discussion\nThese results have significant implications for future work. " * 20
        + "\n\n## Conclusion\nIn conclusion the research demonstrates key findings. " * 10
        + "\n\n![chart](chart.png)\n\nhttps://example.com\nhttps://arxiv.org\nhttps://huggingface.co"
    )
ok3, bonus3 = check_structural(3, topic_dir)
assert ok3
print(f"  Task 3 structural: PASS, bonus={bonus3:.2f}")

# ===================================================================
print("\n=== Full Episode (finalize without LLM) ===")

# Set env vars so LLM grader falls back gracefully
os.environ.pop("API_BASE_URL", None)
os.environ.pop("MODEL_NAME", None)

env2 = ResearcherEnv()
obs = env2.reset(task_number=1, topic_id="topic_01")

# Save a decent analysis file
obs = env2.step(ResearchAction(
    action_type="save_file",
    filename="research_analysis.md",
    content=(
        "# Research Analysis: RAG\n\n"
        "## Background\n"
        + "RAG combines retrieval with generation for better factual accuracy. " * 20
        + "\n\n## Sources\n- https://arxiv.org/abs/2005.11401\n- https://example.com/rag\n- https://docs.llamaindex.ai\n"
        "\n## Candidates\n### Hybrid RAG\nCombines dense and sparse retrieval.\n"
        "### Agentic RAG\nUses agents for multi-step retrieval.\n"
        "### Graph RAG\nUses knowledge graphs.\n"
        "\n## Comparison\n| Approach | Feasibility | Novelty |\n|---|---|---|\n"
        "| Hybrid | High | Medium |\n| Agentic | Medium | High |\n| Graph | Low | High |\n"
        "\n## Selected Topic\nAgentic RAG — best balance of novelty and feasibility.\n"
    ),
))
assert obs.success

# Finalize
obs = env2.step(ResearchAction(action_type="finalize"))
assert obs.done
result = json.loads(obs.result)
assert result["final_reward"] > 0, f"Expected positive reward, got {result}"
print(f"  Finalize OK: reward={result['final_reward']:.3f}, feedback={result.get('feedback', 'n/a')}")

st = env2.state
assert st.is_done
assert st.final_reward is not None
print(f"  State: episode_id={st.episode_id}, steps={st.step_count}, done={st.is_done}")

# ===================================================================
cleanup()
print("\n✅ All smoke tests passed!")
