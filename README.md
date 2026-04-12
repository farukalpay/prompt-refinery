# Prompt Refinery

Retrieval-grounded prompt refinement engine as a reusable Python **library**, production CLI, and **MCP stdio endpoint**.

This project avoids hardcoded prompt templates and static keyword routing. Instead, it derives intent structure from retrieval evidence, then runs one repair/polish generation pass.

## Why this architecture

Most prompt systems break in one of two ways:

1. Overfitted template maps (fast, brittle).
2. Freeform generation (flexible, low control).

Prompt Refinery keeps both control and adaptability:

- Retrieval-first candidate selection from real corpora.
- Data-derived intent spec (objective, audience, locale evidence, slot coverage).
- Single generation call for repair + polish.
- Full runtime artifacts for inspectability (`last_result.json`, SQLite memory).

## Pipeline (research-style view)

```mermaid
flowchart TD
    A["User request"] --> B["Embedding<br/>openai/text-embedding-3-small"]

    B --> C["Prompt candidates<br/>fka/prompts.chat"]
    B --> D["Slot support examples<br/>AmazonScience/massive"]
    B --> E["Memory neighbors<br/>local SQLite"]

    C --> F["Data-driven intent spec<br/>(0 extra LLM calls)"]
    D --> F
    E --> F

    F --> G["Single repair/polish call<br/>mistralai/mistral-nemo"]
    G --> H["Final prompt + persisted runtime evidence"]
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in project root:

```env
OPENROUTER_API_KEY=your_api_key_here
# Optional overrides
# LLM_API_BASE_URL=https://openrouter.ai/api/v1
# EMBED_MODEL=openai/text-embedding-3-small
# REPAIR_MODEL=mistralai/mistral-nemo
```

You can bootstrap from template:

```bash
cp .env.example .env
```

## CLI usage

After install, both commands work:

- `prompt-refinery ...` (console script)
- `python -m prompt_refinery ...` (module entrypoint)
- `./scripts/start_cli.sh ...` (repo-local launcher)

### 1) Default run
Prompt-only output + runtime artifacts.

```bash
prompt-refinery \
  "Write a concise cold email to pitch our AI analytics tool to a logistics startup CEO."
```

### 2) Explicit quality targets (CLI priority)
`--targets` overrides profile/env/default targets.

```bash
prompt-refinery \
  "Design a landing page prompt for B2B fintech onboarding" \
  --targets \
    "Fully specified output" \
    "No unresolved placeholders" \
    "Clear actionable wording"
```

### 3) Custom standards for a domain
Domain-specific quality constraints from CLI.

```bash
prompt-refinery \
  "Create a SOC2 incident response prompt" \
  --targets \
    "Audit-traceable steps" \
    "Owner+deadline per action" \
    "No ambiguous verbs"
```

### 4) JSON output mode
Print full structured result to stdout.

```bash
prompt-refinery \
  "Build a launch checklist prompt" \
  --json
```

## Input -> Output examples

### Featured repaired prompt (full)

**Input prompt**

```text
Create a SOC2 incident response prompt
```

**Quality targets**

```text
Fully specified output
No unresolved placeholders
Clear actionable wording
```

**Output (full repaired prompt)**

```text
ROLE: You are an incident commander and compliance-focused response planner for a B2B SaaS platform.

PRIMARY GOAL:
Create a SOC2-ready incident response runbook that can be executed under pressure and audited later without ambiguity.

OUTPUT FORMAT (MANDATORY):
Return exactly 6 sections in this order.

1) Incident Classification
- Define severity levels Sev-1 to Sev-4.
- For each severity, include: customer impact threshold, security/data impact threshold, escalation owner, and max response start time.

2) First 60 Minutes Timeline
- Provide a minute-by-minute action plan at these checkpoints: T+0, T+5, T+10, T+15, T+30, T+45, T+60.
- Each checkpoint must include:
  - Action
  - Owner role
  - Evidence artifact to capture
  - Hard deadline

3) Containment, Eradication, Recovery
- Containment: immediate risk-reduction actions and rollback criteria.
- Eradication: root-cause elimination tasks and verification steps.
- Recovery: service restoration checklist with validation gates.
- Every task must have an explicit owner and completion criterion.

4) Communication Pack
- Internal update template (engineering + security + leadership).
- Customer-facing status update template.
- Executive summary template for post-incident briefing.
- Include required fields: incident ID, impact scope, current status, next update time, accountable owner.

5) SOC2 Evidence Checklist
- Evidence categories: detection, triage, decision logs, approvals, remediation, customer communication, postmortem actions.
- For each category, specify:
  - Required artifact type
  - System/source of truth
  - Responsible owner role
  - Retention expectation

6) Post-Incident Review and Corrective Actions
- Provide postmortem structure: timeline, root cause, contributing factors, control gaps, corrective actions.
- Corrective actions must include: owner, due date, risk priority, verification method, and closure criteria.

QUALITY CONSTRAINTS:
- No unresolved placeholders.
- No vague verbs such as "handle", "check", or "fix" without measurable criteria.
- Every operational step must be executable by a specific role at a specific time.
- Use concise, direct, and audit-ready language.

STYLE:
Professional, explicit, and operations-focused. Avoid motivational or generic wording.
```

### Full payload files (JSON not inlined here)

- Full sample payloads are in [examples/README.md](./examples/README.md)
- CLI JSON payload sample: [examples/launch_checklist_payload.json](./examples/launch_checklist_payload.json)
- MCP request sample: [examples/mcp_tools_call_request.json](./examples/mcp_tools_call_request.json)
- MCP response sample: [examples/mcp_tools_call_response.json](./examples/mcp_tools_call_response.json)

## Quality-target precedence

Priority order:

1. `--targets` CLI argument
2. `refinery_profile.json`
3. `QUALITY_TARGETS` environment variable (JSON array)
4. Built-in defaults

Default profile (`refinery_profile.json`):

```json
{
  "quality_targets": [
    "Fully specified output",
    "No unresolved placeholders",
    "Clear actionable wording"
  ]
}
```

## Library usage

```python
from pathlib import Path
from prompt_refinery import RefineryEngine, RuntimePaths, RuntimeSettings

project_dir = Path.cwd()
paths = RuntimePaths.from_project_dir(project_dir)
settings = RuntimeSettings.from_env(project_dir)

engine = RefineryEngine(settings=settings, paths=paths)
try:
    result = engine.run(
        user_text="Create a migration playbook prompt for PostgreSQL cutover",
        quality_targets=[
            "Fully specified output",
            "No unresolved placeholders",
            "Rollback steps included"
        ]
    )
    print(result["repaired_prompt"])
finally:
    engine.close()
```

## MCP endpoint (stdio)

### Start server

```bash
prompt-refinery-mcp --project-dir /absolute/path/to/prompt-refinery
```

or with startup script:

```bash
./scripts/start_mcp_stdio.sh --project-dir /absolute/path/to/prompt-refinery
```

### Example MCP client wiring (Claude Desktop / compatible hosts)

```json
{
  "mcpServers": {
    "prompt-refinery": {
      "command": "prompt-refinery-mcp",
      "args": ["--project-dir", "/absolute/path/to/prompt-refinery"]
    }
  }
}
```

### Exposed MCP tool

- `refine_prompt`

Input schema:

- `user_text` (string, required)
- `quality_targets` (string array, optional)
- `export_outputs` (bool, optional, default `true`)

### Example MCP JSON-RPC call (`tools/call`)

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "refine_prompt",
    "arguments": {
      "user_text": "prepare an incident postmortem template for platform outages",
      "quality_targets": [
        "Fully specified output",
        "No unresolved placeholders",
        "Clear actionable wording"
      ],
      "export_outputs": true
    }
  }
}
```

## Local validation snapshot (2026-04-13)

### Test suite

```bash
$ python3 -m pytest
........                                                                 [100%]
8 passed
```

### CLI surface check

```bash
$ python3 -m prompt_refinery --help
usage: prompt-refinery [-h] [--targets T [T ...]] [--profile FILE]
                       [--project-dir DIR] [--json] [--no-gui]
                       [user_text ...]
```

### MCP surface check

```bash
$ python3 -m prompt_refinery.mcp_server --help
usage: prompt-refinery-mcp [-h] [--project-dir PROJECT_DIR]
```

## Optimization notes

Structural changes in this revision:

- Monolithic script moved to importable package (`prompt_refinery/`).
- MASSIVE locale/config coverage is discovered dynamically (no fixed locale seed list).
- Query embedding is now computed **once per request** and reused across prompt/slot/memory retrieval.
- Runtime config and paths are explicit contracts (`RuntimeSettings`, `RuntimePaths`).
- CLI is thin; core logic is library-safe.
- MCP endpoint shares the same engine path as CLI/library (single source of behavior).

## Runtime artifacts

Generated under `runtime_db/`:

- `exports/last_prompt.txt`
- `exports/last_result.json`
- `runtime.sqlite3`
- dataset cache + embedding indices

## Repository layout

```text
prompt_refinery/
  __init__.py
  __main__.py
  cli.py
  core.py
  mcp_server.py
scripts/
  start_cli.sh
  start_mcp_stdio.sh
tests/
  test_intent_spec.py
  test_mcp_server.py
  test_quality_targets.py
refinery_profile.json
.env.example
pyproject.toml
requirements.txt
requirements-dev.txt
LICENSE
README.md
```

## Requirements

- Python 3.10+
- OpenRouter-compatible API key (`OPENROUTER_API_KEY`)

## License

This project is licensed under **GNU General Public License v3.0**.
See [LICENSE](./LICENSE).
