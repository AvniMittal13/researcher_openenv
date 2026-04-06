"""Quick integration test for the HTTP server."""
import json
import requests

BASE = "http://localhost:8000"

# Reset
print("=== RESET ===")
r = requests.post(f"{BASE}/reset", json={"task_number": 1, "topic_id": "topic_01"})
d = r.json()
print(f"done={d['done']}, steps_remaining={d['observation']['steps_remaining']}")

# Step 1
print("\n=== STEP 1: take_notes ===")
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "take_notes", "notes": "RAG is great"}})
d = r.json()
print(f"steps_remaining={d['observation']['steps_remaining']}, reward={d['observation']['cumulative_reward']}")

# Step 2
print("\n=== STEP 2: take_notes ===")
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "take_notes", "notes": "More notes"}})
d = r.json()
print(f"steps_remaining={d['observation']['steps_remaining']}, reward={d['observation']['cumulative_reward']}")

# Step 3: execute_code
print("\n=== STEP 3: execute_code ===")
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "execute_code", "code": "print('hello world')"}})
d = r.json()
print(f"success={d['observation']['success']}, steps_remaining={d['observation']['steps_remaining']}")
print(f"reward={d['observation']['cumulative_reward']}")

# Step 4: save_file
print("\n=== STEP 4: save_file ===")
content = "# Analysis\n\n## Background\n" + "RAG combines retrieval and generation to reduce hallucination in LLMs. " * 30
content += "\n\n## Sources\n- https://example.com - key findings\n- https://arxiv.org - paper\n- https://test.com - tutorial"
content += "\n\n## Candidates\n### Hybrid RAG\nCombines dense and sparse retrieval for better recall and precision.\n"
content += "### Agentic RAG\nUses agents for multi-step retrieval and reasoning.\n"
content += "### Graph RAG\nLeverages knowledge graphs for structured retrieval.\n"
content += "\n## Comparison\n| Approach | Feasibility | Novelty |\n|---|---|---|\n"
content += "| Hybrid | High | Medium |\n| Agentic | Medium | High |\n| Graph | Low | High |\n"
content += "\n## Selected Topic\nAgentic RAG for its strong novelty and practical feasibility balance.\n"
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "save_file", "filename": "research_analysis.md", "content": content}})
d = r.json()
print(f"success={d['observation']['success']}, files={d['observation']['files_created']}")
print(f"steps_remaining={d['observation']['steps_remaining']}, reward={d['observation']['cumulative_reward']}")

# State check
print("\n=== STATE ===")
r = requests.get(f"{BASE}/state")
d = r.json()
print(f"episode_id={d.get('episode_id')}, step_count={d.get('step_count')}")

# Finalize
print("\n=== FINALIZE ===")
r = requests.post(f"{BASE}/step", json={"action": {"action_type": "finalize"}})
d = r.json()
print(f"done={d['done']}, reward={d['observation']['cumulative_reward']}")
print(f"result={d['observation']['result'][:300]}")
