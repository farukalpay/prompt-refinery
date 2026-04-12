# Prompt Refinery

Prompt Refinery is a retrieval-grounded prompt generator that turns short user requests into production-ready prompts.

It combines:
- semantic retrieval over a real prompt corpus (`fka/prompts.chat`)
- slot-aware support examples from MASSIVE (`AmazonScience/massive`)
- session memory reuse for repeated user patterns
- multi-stage model editing for complete, copy-paste-ready output

## Why this project

Most prompt tools either:
- use static templates and break on edge cases, or
- generate from scratch and lose style consistency.

Prompt Refinery keeps both quality and control by selecting a strong base prompt from data, adapting it to the user request, then polishing for clarity and completeness.

## Key capabilities

- **Model-driven retrieval and editing pipeline** (no topic-specific rule tables)
- **Schema-validated dataset ingestion** for both prompts and slot data
- **Robust MASSIVE loading** even when script-based dataset loading is disabled
- **Chunked embeddings** for very long source prompts
- **Intent extraction + polish pass** to better match requested output style at low-to-medium cost
- **Persistent memory** in SQLite for better repeat performance

## Architecture

1. Build local SQLite tables from datasets.
2. Create / load embedding indices for prompts, slot examples, and memory.
3. Retrieve top supports for the user input.
4. Run a low-cost intent extraction model pass.
5. Run the main prompt-repair model pass.
6. Run a low-cost polish pass for completeness and readability.
7. Save outputs + metadata for reuse.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your key in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

Run:

```bash
python3 prompt_refinery.py "Write a concise cold email to pitch our AI analytics tool to a logistics startup CEO."
```

The latest result is written to:
- `runtime_db/exports/last_prompt.txt`
- `runtime_db/exports/last_result.json`

## Configuration

Environment variables:

- `OPENAI_API_KEY` (required)
- `LLM_API_BASE_URL` (optional, default: `https://openrouter.ai/api/v1`)
- `EMBED_MODEL` (optional)
- `REPAIR_MODEL` (optional)
- `INTENT_MODEL` (optional)
- `POLISH_MODEL` (optional)

Default profile is tuned for low-to-medium cost:
- embedding: `openai/text-embedding-3-small`
- repair: `mistralai/mistral-nemo`
- intent/polish: `openai/gpt-4o-mini`

## Example (full output style)

Input:

```text
Write a concise cold email to pitch our AI analytics tool to a logistics startup CEO.
```

Output style (fully written, not truncated, no placeholder fragments):

```text
ROLE: Act as an "A-List" Direct Response Copywriter (Gary Halbert or David Ogilvy style).

GOAL: Write a cold email to a logistics startup CEO with the objective of selling our AI analytics tool.
CLIENT PROBLEM: Manual reporting delays decisions, hides route-level inefficiencies, and inflates operating cost.
MY SOLUTION: An AI analytics platform that unifies shipment, route, and fulfillment data into real-time decision support.

EMAIL ENGINEERING:

Subject Line: Generate 5 options that create immediate curiosity or clear business benefit.

The Hook: The first sentence must be a pattern interrupt that demonstrates concrete understanding of logistics operations.

The Value Proposition (The Meat): Connect the pain point to the solution using a "Before vs. After" structure.

Objection Handling: Defuse expected concerns around implementation time and budget.

CTA (Call to Action): Use a low-friction next step (e.g., "Are you open to a 5-minute walkthrough video this week?").

TONE: Professional, conversational, confident, brief (under 150 words).
```

## Repository layout

```text
prompt_refinery.py      # main application
requirements.txt        # Python dependencies
.env.example            # sample local configuration
runtime_db/             # local runtime artifacts (gitignored)
```

## Notes

- This project stores runtime data locally and keeps secrets out of Git with `.gitignore`.
- If you run from a desktop app without CLI args, a small GUI input/output fallback is available.
