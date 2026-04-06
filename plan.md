# ResearcherEnv — Detailed Execution Plan

## 1. What We're Building

An OpenEnv environment where an LLM agent autonomously researches topics by searching the web, scraping pages (including arxiv PDFs), executing Python code (for visualizations), taking notes, and producing a structured markdown report. The environment exposes 5 tools via the `step()` action space, grades reports with a 3-layer system (structural + keyword + LLM-judge), and runs inside a Docker container on HF Spaces.

---

## 2. File Structure (Following the 3-Component Pattern from SPAG)

```
researcher1/
├── plan.md                    ← This file
├── requirements.txt           ← Python dependencies
├── Dockerfile                 ← For HF Spaces deployment
├── env/
│   ├── __init__.py
│   ├── models.py              ← Pydantic: ResearchAction, ResearchObservation, ResearchState
│   └── researcher_env.py      ← Core environment: reset(), step(), state
├── graders/
│   ├── __init__.py
│   └── research_grader.py     ← 3-layer grading: structural + keyword + LLM-judge
├── server/
│   ├── __init__.py
│   └── app.py                 ← FastAPI server (create_fastapi_app)
├── tools/
│   ├── __init__.py
│   ├── web_search.py          ← DuckDuckGo search wrapper
│   ├── scraper.py             ← HTML + PDF scraping (bs4 + PyMuPDF)
│   └── code_executor.py       ← Sandboxed subprocess Python execution
├── tasks/
│   ├── __init__.py
│   └── task_registry.py       ← 3 tasks: analysis → experimentation → report (+ grading configs)
├── inference.py               ← Agent policy using OpenAI client (required format)
├── test_smoke.py              ← Smoke tests for graders + env
└── data/
    └── topics.json            ← Research areas (used across all 3 tasks)
```

---

## 3. Pydantic Models (`env/models.py`)

### ResearchAction
```python
class ResearchAction(Action):
    action_type: Literal["web_search", "scrape_url", "execute_code", "take_notes", "save_file", "finalize"]
    query: Optional[str] = None        # For web_search
    url: Optional[str] = None          # For scrape_url
    code: Optional[str] = None         # For execute_code
    notes: Optional[str] = None        # For take_notes
    filename: Optional[str] = None     # For save_file (e.g., "research_analysis.md", "report.md")
    content: Optional[str] = None      # For save_file (file content to write)
```

### ResearchObservation
```python
class ResearchObservation(Observation):
    action_type: str                    # Echo back which action was taken
    success: bool                       # Did the action succeed?
    result: str                         # Action-specific output (search results, scraped text, stdout, etc.)
    files_created: list[str]            # Files written to /output/ (from execute_code)
    error: Optional[str] = None         # Error message if action failed
    steps_remaining: int                # max_steps - steps_taken
    cumulative_reward: float            # Running partial reward
```

### ResearchState
```python
class ResearchState(State):
    task_id: str
    task_number: int                     # 1, 2, or 3 — which task in the pipeline
    research_question: str
    search_results: list[dict]          # [{query, results: [{title, url, snippet}]}]
    scraped_pages: list[dict]           # [{url, type, content_preview}]
    notes: str                          # Accumulated notes from take_notes
    files_created: list[str]            # Files in /output/ (research_analysis.md, report.md, etc.)
    code_history: list[dict]            # [{code, stdout, stderr, exit_code}]
    steps_taken: int
    max_steps: int                      # 20
    done: bool
    final_reward: Optional[float]       # Set after finalize
```

---

## 4. Environment Logic (`env/researcher_env.py`)

### `reset(seed, episode_id, task_id, **kwargs) -> ResearchObservation`
1. Look up task from `task_registry` by `task_id` (or pick random if not specified)
2. Initialize empty `ResearchState` with the task's research question, difficulty, max_steps=20
3. Create a fresh `/output/` directory for this episode
4. Return observation with the research question as `result`

### `step(action: ResearchAction) -> ResearchObservation`
Dispatch based on `action.action_type`:

| Action | What happens | Observation.result |
|--------|-------------|-------------------|
| `web_search` | Call `duckduckgo_search` with `action.query`, store in state | JSON of top 5 results `[{title, url, snippet}]` |
| `scrape_url` | Fetch URL, detect HTML vs PDF, extract text (cap 8000 chars), store in state | Extracted text content |
| `execute_code` | Run `action.code` via subprocess (timeout=30s), collect files from `/output/` | `{stdout, stderr, exit_code, files}` |
| `take_notes` | Append `action.notes` to `state.notes` | Confirmation + current notes length |
| `save_file` | Write `action.content` to `/output/{action.filename}`, add to state.files_created | Confirmation + filename |
| `finalize` | Run the task-specific LLM grader on the relevant output file, set done=True | `{reward, grading_breakdown}` |

**Partial rewards**: After each step, compute lightweight running reward:
- +0.05 per `web_search` call (cap at 0.15)
- +0.05 per `scrape_url` call (cap at 0.15)
- +0.03 per `execute_code` with exit_code=0 (cap at 0.09)
- +0.05 per `take_notes` call (cap at 0.1)

These partial rewards encourage tool usage but the bulk (0.51+) comes from `finalize`.

**Episode termination**: `done = True` when agent calls `finalize` OR `steps_taken >= max_steps`.

---

## 5. Tools Implementation (`tools/`)

### `web_search.py`
```python
from duckduckgo_search import DDGS

def search(query: str, max_results: int = 5) -> list[dict]:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return [{"title": r["title"], "url": r["href"], "snippet": r["body"]} for r in results]
```

### `scraper.py`
- Detect content type from response headers + URL suffix
- **HTML path**: `requests.get()` → `BeautifulSoup` → strip scripts/nav/footer → `.get_text()[:8000]`
- **PDF path**: `requests.get()` → `fitz.open(stream=content)` → extract text from first 12 pages → `[:8000]`
- **arxiv special case**: If URL matches `arxiv.org/abs/XXXX`, try `ar5iv.org/html/XXXX` first (cleaner text)
- **Timeout**: 30s on all requests
- **User-Agent**: `research-agent/1.0`

### `code_executor.py`
```python
def execute(code: str, output_dir: str, timeout: int = 30) -> dict:
    preamble = f"import os\nOUTPUT_DIR = '{output_dir}'\nos.makedirs(OUTPUT_DIR, exist_ok=True)\n"
    full_code = preamble + code
    result = subprocess.run(
        ["python", "-c", full_code],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "MPLBACKEND": "Agg"}  # Non-interactive matplotlib
    )
    files = os.listdir(output_dir) if os.path.exists(output_dir) else []
    return {
        "stdout": result.stdout[:3000],
        "stderr": result.stderr[:500],
        "exit_code": result.returncode,
        "files": files
    }
```
**Security**: Set `MPLBACKEND=Agg` to prevent display calls. The subprocess timeout prevents runaway code. Code runs inside the Docker container anyway (sandboxed).

---

## 6. Task Registry (`tasks/task_registry.py`)

The 3 tasks form a **sequential research pipeline**. Each task builds on the previous one's output. A single episode runs one task; the agent is told which task number to execute.

---

### Task 1 — "Topic Summary, Analysis & Candidate Comparison"

**Goal**: Research a broad area, identify candidate sub-topics, compare them, and select the most promising one for deeper investigation.

**Agent instructions**: 
> Search the web for the given research area. Scrape at least 3 relevant sources. Identify 3-5 candidate research topics/approaches within the area. Compare them on feasibility, novelty, and impact. Select one for deeper investigation. Save your complete analysis to `research_analysis.md`.

**Expected output file**: `/output/research_analysis.md` (saved via `save_file` action)

**Expected structure of `research_analysis.md`**:
```markdown
# Research Analysis: {area}

## 1. Background & Context
...(summary of the area, key findings from sources)...

## 2. Sources Reviewed
- [Source 1](url) — key takeaway
- [Source 2](url) — key takeaway
...

## 3. Candidate Topics
### 3.1 Topic A
- Description, pros, cons, feasibility
### 3.2 Topic B
...
### 3.3 Topic C
...

## 4. Comparison Table
| Topic | Feasibility | Novelty | Impact | Overall |
|-------|------------|---------|--------|--------|
| A     | ...        | ...     | ...    | ...    |

## 5. Selected Topic & Justification
...(which topic and why)...
```

**LLM Grader prompt for Task 1**:
```
You are grading a research analysis document. The research area was: "{research_question}"

The document should contain:
1. A clear background summary with cited sources (0-0.2)
2. At least 3 candidate topics identified with pros/cons (0-0.25)
3. A structured comparison (table or detailed prose) (0-0.2)
4. A justified selection of one topic for further research (0-0.15)
5. Overall quality: clarity, depth, accuracy (0-0.2)

Document:
{content}

Respond with JSON: {"score": 0.X, "breakdown": {"background": X, "candidates": X, "comparison": X, "selection": X, "quality": X}, "feedback": "one sentence"}
```

---

### Task 2 — "Experimentation & Coding"

**Goal**: Take the selected topic from Task 1 and conduct hands-on experimentation — write and run code, produce experimental results, generate visualizations.

**Agent instructions**:
> Read the `research_analysis.md` from Task 1 to understand the selected topic. Write Python code to explore/experiment with the topic. This could be: running a benchmark, implementing an algorithm, analyzing a dataset, or comparing approaches programmatically. Execute the code, collect results. Generate at least one visualization (chart/plot). Save all code and results. Your experimental outputs should be in `/output/` (plots, data files, etc.).

**Expected outputs in `/output/`**:
- Python scripts executed via `execute_code` (tracked in `code_history`)
- Generated plots/charts (`.png` files)
- Optional: intermediate data files (`.csv`, `.json`)
- The `research_analysis.md` from Task 1 is assumed to already exist (passed as context)

**LLM Grader prompt for Task 2**:
```
You are grading a coding and experimentation session for a research project.
Research topic: "{selected_topic}"

Evaluate based on:
1. Code quality: Does the code run successfully? Is it relevant to the topic? (0-0.25)
2. Experimental rigor: Are the experiments meaningful, not just trivial? (0-0.25)
3. Results produced: Are there concrete outputs (numbers, tables, data)? (0-0.2)
4. Visualizations: Is at least one chart/plot generated and relevant? (0-0.15)
5. Reproducibility: Could someone re-run this and get similar results? (0-0.15)

Code history:
{code_history_summary}

Files created: {files_list}
Stdout outputs: {stdout_summary}

Respond with JSON: {"score": 0.X, "breakdown": {"code_quality": X, "rigor": X, "results": X, "visualizations": X, "reproducibility": X}, "feedback": "one sentence"}
```

---

### Task 3 — "Complete Research Report"

**Goal**: Synthesize everything from Tasks 1 and 2 into a polished, complete research report.

**Agent instructions**:
> You have access to `research_analysis.md` (from Task 1) and the experimental results/code/plots (from Task 2). Write a complete, well-structured research report that combines the literature review, experimental methodology, results, and conclusions. Save the final report as `report.md` in `/output/`.

**Expected output file**: `/output/report.md` (saved via `save_file` action)

**Expected structure of `report.md`**:
```markdown
# Research Report: {topic}

## Abstract
...(100-150 word summary)...

## 1. Introduction
...(background, motivation, research question)...

## 2. Literature Review
...(sources, prior work, context — draws from research_analysis.md)...

## 3. Methodology
...(what experiments were run, how, why)...

## 4. Results
...(experimental findings, tables, references to plots)...
![Chart](llm_growth.png)

## 5. Discussion
...(interpretation, limitations, comparison to prior work)...

## 6. Conclusion
...(summary, future directions)...

## References
- [1] ...
- [2] ...
```

**LLM Grader prompt for Task 3**:
```
You are grading a complete research report. The topic is: "{selected_topic}"

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

Respond with JSON: {"score": 0.X, "breakdown": {"structure": X, "literature": X, "methodology": X, "results": X, "analysis": X, "writing": X, "completeness": X}, "feedback": "one sentence"}
```

---

### Task Topics (`data/topics.json`)

Each entry defines a research area. The same area is used across all 3 tasks in a pipeline run.

```json
[
  {
    "topic_id": "topic_01",
    "research_area": "Retrieval-Augmented Generation (RAG) for domain-specific applications",
    "expected_keywords": ["RAG", "retrieval", "vector database", "embedding", "chunking", "hallucination"]
  },
  {
    "topic_id": "topic_02",
    "research_area": "Parameter-efficient fine-tuning methods for large language models",
    "expected_keywords": ["LoRA", "QLoRA", "adapter", "fine-tuning", "PEFT", "parameters"]
  },
  {
    "topic_id": "topic_03",
    "research_area": "LLM evaluation benchmarks and their limitations",
    "expected_keywords": ["benchmark", "MMLU", "HumanEval", "leaderboard", "contamination", "evaluation"]
  },
  {
    "topic_id": "topic_04",
    "research_area": "Vision-language models and multimodal AI",
    "expected_keywords": ["CLIP", "multimodal", "vision", "image-text", "contrastive", "VLM"]
  },
  {
    "topic_id": "topic_05",
    "research_area": "Reinforcement learning from human feedback (RLHF) alternatives",
    "expected_keywords": ["RLHF", "DPO", "preference", "alignment", "reward model", "KTO"]
  }
]
```

5 topics. Each topic can be used for Task 1, 2, or 3 independently.

---

## 7. Grading System (`graders/research_grader.py`)

Each task has its own dedicated LLM grader. Grading is a **2-layer** process:

### Layer 1 — Structural Pre-checks (deterministic, fast, no LLM)

Quick sanity checks before invoking the LLM. If the output file doesn't exist or is nearly empty, return reward=0.0 immediately.

**Task 1 checks** (on `research_analysis.md`):
- File exists → required
- Word count >= 200 → required
- At least 2 markdown headers → required
- At least 2 URLs cited → bonus

**Task 2 checks** (on code execution history):
- At least 1 `execute_code` call with exit_code=0 → required
- At least 1 file generated (plot/data) → required

**Task 3 checks** (on `report.md`):
- File exists → required
- Word count >= 400 → required
- At least 4 markdown headers → required
- At least 1 image reference `![...](...)`→ bonus
- At least 3 URLs cited → bonus

If required checks fail → reward = 0.0, skip LLM grading.

### Layer 2 — LLM-as-Judge (per-task rubric)

Each task uses the **task-specific LLM prompt** defined in Section 6 above. The LLM returns a JSON with `score` (0.0–1.0) and a `breakdown` dict.

- Use the model at `API_BASE_URL` via OpenAI client
- Timeout: 15s
- Fallback: if LLM call fails or returns invalid JSON, use 0.4 as default
- Content fed to LLM is capped at 3000 chars to stay fast

### Final Reward Computation
```python
def grade_task(task_number: int, state: ResearchState, output_dir: str) -> GradeResult:
    # Layer 1: structural pre-checks
    structural_ok, structural_bonus = check_structural(task_number, state, output_dir)
    if not structural_ok:
        return GradeResult(reward=0.0, breakdown={"reason": "structural checks failed"})

    # Layer 2: LLM grader (task-specific prompt)
    content = read_task_output(task_number, state, output_dir)
    llm_result = call_llm_grader(task_number, content, state)

    # Combine: LLM score (0.85 weight) + structural bonus (0.15 weight)
    final = llm_result["score"] * 0.85 + structural_bonus * 0.15
    return GradeResult(reward=min(final, 1.0), breakdown=llm_result["breakdown"])
```

**What the LLM grader sees per task**:
| Task | Input to grader | Rubric focus |
|------|----------------|-------------|
| 1 | `research_analysis.md` content | Background, candidates, comparison, selection, quality |
| 2 | Code history + stdout + files list | Code quality, rigor, results, visualizations, reproducibility |
| 3 | `report.md` content | Structure, literature, methodology, results, analysis, writing, completeness |

---

## 8. Server (`server/app.py`)

```python
from openenv.core.env_server.http_server import create_fastapi_app
from researcher1.env.models import ResearchAction, ResearchObservation
from researcher1.env.researcher_env import ResearcherEnv

app = create_fastapi_app(
    env=ResearcherEnv,
    action_cls=ResearchAction,
    observation_cls=ResearchObservation,
)
```

Also mount `/files/` as a static route to serve generated plots/images from `/output/`.

---

## 9. Inference Script (`inference.py`)

**Must follow exact hackathon format**: uses OpenAI client, `[START]`/`[STEP]`/`[END]` log format.

```python
# Pseudocode structure
from openai import OpenAI

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# --- Task-specific system prompts ---

TASK_1_PROMPT = """You are a research agent. Your task is to research a topic area, identify candidate sub-topics, compare them, and produce a research analysis.

Tools available (respond with ONE JSON action per step):
- {"action_type": "web_search", "query": "..."} — Search the web
- {"action_type": "scrape_url", "url": "..."} — Read a webpage or PDF
- {"action_type": "take_notes", "notes": "..."} — Accumulate research notes
- {"action_type": "save_file", "filename": "research_analysis.md", "content": "..."} — Save a file
- {"action_type": "finalize"} — Submit for grading (call when done)

Workflow:
1. Search for the research area to understand the landscape (2-3 searches)
2. Scrape 3-5 of the most relevant URLs
3. Take notes on key findings
4. Identify 3-5 candidate research topics within this area
5. Compare them on feasibility, novelty, and impact
6. Select the most promising topic and justify your choice
7. Save everything to research_analysis.md with sections: Background, Sources, Candidate Topics, Comparison Table, Selected Topic
8. Call finalize
"""

TASK_2_PROMPT = """You are a research agent. Your task is to conduct coding experiments on a selected research topic.

The file research_analysis.md contains your prior analysis with a selected topic. Read it first.

Tools available (respond with ONE JSON action per step):
- {"action_type": "web_search", "query": "..."} — Search for code examples, datasets, APIs
- {"action_type": "scrape_url", "url": "..."} — Read documentation or tutorials
- {"action_type": "execute_code", "code": "..."} — Run Python code (OUTPUT_DIR is pre-set; save plots/data there)
- {"action_type": "take_notes", "notes": "..."} — Record observations
- {"action_type": "finalize"} — Submit for grading (call when done)

Workflow:
1. Read research_analysis.md context (provided in observation)
2. Search for relevant code examples, datasets, or libraries
3. Write and execute Python code to experiment with the topic (benchmarks, comparisons, implementations)
4. Generate at least one visualization (matplotlib chart saved to OUTPUT_DIR)
5. Execute multiple experiments if needed — iterate on results
6. Take notes on findings and results
7. Call finalize
"""

TASK_3_PROMPT = """You are a research agent. Your task is to write a complete, polished research report.

You have access to research_analysis.md (literature review) and experimental results (code outputs, plots) from prior tasks.

Tools available (respond with ONE JSON action per step):
- {"action_type": "web_search", "query": "..."} — Search for additional context if needed
- {"action_type": "scrape_url", "url": "..."} — Read additional sources if needed
- {"action_type": "execute_code", "code": "..."} — Run code for additional analysis/plots if needed
- {"action_type": "take_notes", "notes": "..."} — Draft sections
- {"action_type": "save_file", "filename": "report.md", "content": "..."} — Save the final report
- {"action_type": "finalize"} — Submit for grading (call when done)

Workflow:
1. Review the research_analysis.md and experimental outputs (provided in observation)
2. Draft and save report.md with these sections: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Conclusion, References
3. Reference generated plots using ![caption](filename.png)
4. Ensure the report is comprehensive (500+ words) with proper citations
5. Call finalize
"""

TASK_PROMPTS = {1: TASK_1_PROMPT, 2: TASK_2_PROMPT, 3: TASK_3_PROMPT}

def run_episode(env_url, task_number, topic_id):
    print("[START]")
    obs = reset(env_url, task_number=task_number, topic_id=topic_id)
    system_prompt = TASK_PROMPTS[task_number]

    while not obs["done"]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research area: {obs['research_question']}\n\nCurrent observation:\n{obs['result']}"}
        ]
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )
        action = parse_action(response.choices[0].message.content)
        print(f"[STEP] task={task_number} action={action['action_type']}")
        obs = step(env_url, action)

    print(f"[END] task={task_number} reward={obs['cumulative_reward']}")

# Full pipeline: run all 3 tasks sequentially for a topic
def run_pipeline(env_url, topic_id):
    for task_num in [1, 2, 3]:
        run_episode(env_url, task_number=task_num, topic_id=topic_id)
```

---

## 10. Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# System dependencies for PyMuPDF and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libmupdf-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY researcher1/ researcher1/

EXPOSE 8000

CMD ["uvicorn", "researcher1.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 11. Dependencies (`requirements.txt`)

```
openenv-core>=0.2.2
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
duckduckgo-search>=7.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
PyMuPDF>=1.24.0
matplotlib>=3.8.0
numpy>=1.26.0
openai>=1.0.0
```

---

## 12. Execution Timeline

### Phase 1: Scaffold & Models (~ first chunk of work)
1. Create all directories and `__init__.py` files
2. Write `env/models.py` — the Pydantic types
3. Write `data/topics.json` — all 15 research tasks
4. Write `requirements.txt`

### Phase 2: Tools (build bottom-up)
5. Write `tools/web_search.py` — DuckDuckGo wrapper
6. Write `tools/scraper.py` — HTML + PDF scraping
7. Write `tools/code_executor.py` — sandboxed subprocess execution
8. Test each tool standalone in a quick script

### Phase 3: Graders
9. Write `graders/research_grader.py` — all 3 layers
10. Write unit tests for grading (known inputs → expected scores)

### Phase 4: Environment Core
11. Write `env/researcher_env.py` — `reset()`, `step()`, `state`
12. Wire up tools → step dispatch
13. Wire up graders → finalize action
14. Write `test_smoke.py` — end-to-end episode test

### Phase 5: Server & Local Testing
15. Write `server/app.py`
16. Run locally with `uvicorn`, test `/health`, `/reset`, `/step`
17. Run a manual episode via curl/httpie

### Phase 6: Inference Script
18. Write `inference.py` with exact `[START]`/`[STEP]`/`[END]` format
19. Test against local server
20. Run pre-validation script

### Phase 7: Docker & Deploy
21. Write `Dockerfile`
22. Build and test locally: `docker build -t researcher-env . && docker run -p 8000:8000 researcher-env`
23. Push to HF Spaces: `openenv push --repo-id <username>/researcher-env`
24. Verify Space returns 200 on `/health` and `/reset`

### Phase 8: Polish & Submit
25. Run the hackathon pre-validation script
26. Test full inference loop against deployed Space
27. Write README with examples
28. Submit

---

## 13. Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Web search rate-limited | Cache results in state; DuckDuckGo is lenient |
| Scraping fails (JS-heavy sites) | Fallback to snippet from search results; ar5iv for arxiv |
| Code execution timeout | 30s hard limit via subprocess; agent sees error and can retry |
| LLM judge too slow | 10s timeout, fallback to 0.5 score |
| Total episode > 20 min | max_steps=20, each tool capped at 30s = worst case ~10 min |
| PDF parsing fails | PyMuPDF fallback; return error obs so agent can try HTML |
| Docker build fails on HF | Use `python:3.12-slim`, pre-test locally |
| Network disabled in Docker | Must enable networking in HF Space config (required for search/scrape) |

---

## 14. What Makes This Stand Out

- **6 tool types in one episode** (search, scrape, code, notes, save_file, finalize) — most environments have 1-2
- **3-task sequential pipeline** — analysis → experimentation → report, each graded independently by LLM
- **Partial rewards per step** — not just 0/1 at the end
- **Multi-turn reasoning** — agent must plan, gather, synthesize, experiment, produce
- **Real-world task** — not a game; directly useful as a research assistant
- **Dedicated LLM grader per task** — rubric-based evaluation tailored to each phase
- **arxiv-aware scraping** — handles academic PDFs natively
- **Code-generated visualizations** — agent writes matplotlib code, env executes it
- **Progressive output** — `research_analysis.md` feeds into experimentation, which feeds into `report.md`