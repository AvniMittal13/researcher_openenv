---
title: ResearcherEnv
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# ResearcherEnv — OpenEnv Research Agent Environment

An OpenEnv environment where an LLM agent autonomously researches topics by searching the web, scraping pages (including academic PDFs), executing Python code for experiments and visualizations, and producing structured deliverables.

## Environment Overview

**ResearcherEnv** implements a 3-task sequential research pipeline:

| Task | Name | Goal | Output |
|------|------|------|--------|
| 1 | Topic Analysis | Research a broad area, identify candidate sub-topics, compare them, select one | `research_analysis.md` |
| 2 | Experimentation | Write and run code to experiment with the selected topic, generate visualizations | Code outputs + plots |
| 3 | Research Report | Synthesize everything into a polished research report | `report.md` |

Each task is a separate episode with its own grading rubric. Tasks build on each other — Task 2 reads Task 1's output, Task 3 reads both.

## Action Space

The agent selects one action per step:

```python
class ResearchAction(Action):
    action_type: Literal["web_search", "scrape_url", "execute_code", "take_notes", "save_file", "finalize"]
    query: Optional[str]      # For web_search
    url: Optional[str]        # For scrape_url
    code: Optional[str]       # For execute_code
    notes: Optional[str]      # For take_notes
    filename: Optional[str]   # For save_file
    content: Optional[str]    # For save_file
```

| Action | Description |
|--------|-------------|
| `web_search` | Search the web via DuckDuckGo, returns top 5 results |
| `scrape_url` | Fetch and extract text from a URL (HTML or PDF) |
| `execute_code` | Run Python code in a sandboxed subprocess |
| `take_notes` | Accumulate research notes in the episode state |
| `save_file` | Write a file to the output directory |
| `finalize` | Trigger grading and end the episode |

## Observation Space

```python
class ResearchObservation(Observation):
    action_type: str             # Which action was taken
    success: bool                # Did it succeed?
    result: str                  # Action output (search results, scraped text, stdout, etc.)
    files_created: list[str]     # Files in output directory
    error: Optional[str]         # Error message if failed
    steps_remaining: int         # Steps left before auto-finalize
    cumulative_reward: float     # Running partial reward
```

## State

```python
class ResearchState(State):
    task_number: int             # 1, 2, or 3
    topic_id: str
    research_question: str
    search_results: list[dict]
    scraped_pages: list[dict]
    notes: str
    files_created: list[str]
    code_history: list[dict]
    max_steps: int               # 20
    is_done: bool
    final_reward: Optional[float]
```

## Reward Function

### Partial Rewards (per step)
- `web_search`: +0.05 each (max 3)
- `scrape_url`: +0.05 each (max 3)
- `execute_code` (success): +0.03 each (max 3)
- `take_notes`: +0.05 each (max 2)
- `save_file`: +0.02 each (max 2)

### Final Grading (on `finalize`)
Two-layer system:
1. **Structural checks** (deterministic): file exists, word count, headers, citations
2. **LLM-as-judge** (per-task rubric): evaluates quality, depth, accuracy

Final reward = LLM score × 0.85 + structural bonus × 0.15, capped at 1.0.

## Tasks & Topics

5 research areas, each usable across all 3 tasks:
- Retrieval-Augmented Generation (RAG)
- Parameter-efficient fine-tuning (LoRA/QLoRA)
- LLM evaluation benchmarks
- Vision-language models
- RLHF alternatives (DPO/KTO)

## Setup

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Install & Run Locally
```bash
cd researcher1
uv sync
uv run uvicorn server.app:app --port 8000
```

### Run with Docker
```bash
cd researcher1
docker build -t researcher-env .
docker run -p 8000:8000 \
  -e API_BASE_URL="https://api.groq.com/openai/v1" \
  -e MODEL_NAME="llama-3.1-8b-instant" \
  -e HF_TOKEN="your-api-key" \
  researcher-env
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | API key for the LLM provider |

### Run Inference
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your-key"
python inference.py --env-url http://localhost:8000 --topic topic_01 --tasks 1,2,3
```

### Validate
```bash
openenv validate .          # Local structure check
openenv validate --url http://localhost:8000  # Runtime check
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute one action |
| `/state` | GET | Get current episode state |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/docs` | GET | Swagger UI |
