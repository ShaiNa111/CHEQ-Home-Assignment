# Customer Churn Analytics — MCP Server

> Ask natural-language questions about the Telco Customer Churn dataset and get accurate,
> SQL-backed answers — directly inside Claude Code or Codex.

---

## What it does

This MCP server loads the **Customer Churn dataset**
([`aai510-group1/telco-customer-churn`](https://huggingface.co/datasets/aai510-group1/telco-customer-churn)
on Hugging Face) into an in-memory [DuckDB](https://duckdb.org/) database and exposes six tools
that Claude (or any MCP client) can call to answer business questions:

| Tool | Purpose |
|------|---------|
| **`ask_question(question)`** | **Primary tool.** NL → SQL via a local Ollama model — pass any question directly, no schema lookup needed *(free, no API key)* |
| `get_schema()` | Power-user: column names + types, for writing raw SQL |
| `get_sample_rows(n)` | Power-user: inspect actual values and encoding |
| `query_sql(sql)` | Power-user: run any `SELECT` / `WITH` query directly |
| `get_churn_summary()` | Pre-computed overview: overall rate, by contract, payment method, revenue |
| `get_column_distribution(column)` | Value counts (categorical) or descriptive stats (numeric) |

**`ask_question` is the intended default path** — it runs a dedicated local LLM that
converts your question straight into SQL and returns the answer, demonstrating the
server's own model integration (per the assignment's "Model/API" requirement) rather
than relying solely on whichever LLM happens to be orchestrating the MCP session. The
other five tools are lower-level building blocks: useful as a fallback if Ollama isn't
running, or for power users who want to write/inspect SQL directly.

Claude Code reads the tool descriptions, calls `ask_question` first for natural-language
queries, and falls back to the other tools (or hand-written SQL via `query_sql`) only if
needed.

---

## Prerequisites

| Requirement | Minimum version |
|-------------|----------------|
| Python | 3.11+ |
| pip | any recent |
| Claude Code | latest |
| [Ollama](https://ollama.com) | latest — *optional, only for the `ask_question` tool* |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/ShaiNa111/CHEQ-Home-Assignment.git
cd CHEQ-Home-Assignment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or, with `uv` (faster):

```bash
uv pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python3 setup_data.py
```

Downloads the dataset from Hugging Face (`aai510-group1/telco-customer-churn`) via the
`datasets` library and saves it to `data/churn.csv`. Requires internet access and the
`datasets` package (included in `requirements.txt`).

After setup, `data/churn.csv` should exist.

> **Note on column names:** this dataset uses spaced column names like
> `"Tenure in Months"`, `"Monthly Charge"`, `"Internet Type"`, `"Payment Method"`.
> The churn target is a single `"Churn"` column, already encoded as `0`/`1` (BIGINT) —
> there is no separate "Churn Label"/"Churn Value" split. Most other boolean-like
> columns (`Partner`, `Dependents`, `Senior Citizen`, `Phone Service`, `Paperless Billing`,
> `Online Security`, `Streaming TV`, etc.) are **also** `0`/`1` BIGINT, not `'Yes'`/`'No'`
> strings — run `get_sample_rows()` or `get_schema()` if you're unsure how a column is
> encoded before filtering on it. The server's tools (`get_churn_summary`,
> `get_column_distribution`) handle this automatically; when writing raw SQL via
> `query_sql`, just remember to wrap multi-word column names in double quotes, e.g.
> `WHERE "Churn" = 1` (not `'Yes'`).

### 4. Set up Ollama (recommended — powers the primary `ask_question` tool)

`ask_question` is the primary, recommended way to query this server — it runs a
dedicated local LLM that converts your question into SQL directly, so the server
demonstrates its own model integration rather than relying solely on whichever LLM
happens to be orchestrating the MCP session. It runs **completely free and locally**
via [Ollama](https://ollama.com) — no API key, no account, no cost.

```bash
# 1. Install Ollama (macOS / Linux / Windows)
brew install ollama          # macOS, or download from https://ollama.com

# 2. Pull a SQL-capable model (recommended for Apple Silicon, ~9GB)
ollama pull qwen2.5-coder:14b

# 3. Ollama runs automatically as a background service on localhost:11434
#    Verify it's up:
curl http://localhost:11434/api/tags
```

That's it — no environment variables required for the defaults. If you want to use a
different model or a remote Ollama instance, set:

```bash
export OLLAMA_MODEL="qwen2.5-coder:32b"          # default: qwen2.5-coder:14b
export OLLAMA_BASE_URL="http://localhost:11434"  # default shown
```

If you'd rather skip this, the other five tools (`get_schema`, `get_sample_rows`,
`query_sql`, `get_churn_summary`, `get_column_distribution`) need **no LLM of their
own** — Claude Code's own model can drive the natural-language reasoning through them
instead, as a fallback.

**Model recommendation:** `qwen2.5-coder:14b` is the sweet spot for text-to-SQL on most
consumer machines (Apple Silicon with 36GB+ unified memory runs it comfortably and fast).
If you have 48GB+ memory, `qwen2.5-coder:32b` gives a further accuracy bump. Qwen2.5-Coder
was chosen over general-purpose models (Llama, Mistral) because it's specifically trained
on code/SQL and outperforms larger generalist models on SQL benchmarks at this size.

If Ollama isn't installed or running, `ask_question` returns a clear setup message instead
of failing silently — and `query_sql` still works for writing SQL by hand.

### 5. Verify everything works

Run the full verification suite — it checks all 7 tools end-to-end, including
`ask_question` if Ollama is running (skipped gracefully otherwise):

```bash
python3 test_server.py
```

Expected output (with Ollama running):

```
✓ [PASS] import server.py
✓ [PASS] get_schema()
✓ [PASS] get_sample_rows(3)
✓ [PASS] query_sql() — valid SELECT
✓ [PASS] query_sql() — blocks non-SELECT
✓ [PASS] get_churn_summary()
✓ [PASS] get_column_distribution('Contract')
✓ [PASS] ask_question() — NL to SQL via Ollama

8 passed, 0 failed, 0 skipped (of 8 checks)
✓ All checks passed, including ask_question via Ollama.
```

This also doubles as a quick sanity check after cloning — if it passes, the
server is ready to connect to Claude Code.

> **Note:** `test_server.py` calls the tool functions directly (no MCP protocol
> overhead). Claude Code's own `/run` verification feature tests the server
> differently — it launches `server.py` as a real subprocess and drives it
> through the actual MCP protocol (`initialize`, `list tools`, `call tool`),
> the same way Claude Code does in production. Either is a valid way to verify
> the server works; neither one registers the server for ongoing use — that's
> a separate step, see below.

---

## Connect to Claude Code

**Recommended: use the `claude mcp add` CLI command.** This is the officially supported
way to register an MCP server — it creates/edits the config file for you, so you don't
need to find or hand-edit any JSON.

```bash
cd CHEQ-Home-Assignment
claude mcp add churn-analytics -- python3 "$(pwd)/server.py"
```

Verify it registered:

```bash
claude mcp list
```

You should see `churn-analytics` in the list. Restart Claude Code if it was already
running, then ask it something like *"use the churn-analytics tools to find the overall
churn rate."*

If you've set up Ollama with a non-default model, pass it as an environment variable:

```bash
claude mcp add churn-analytics --env OLLAMA_MODEL=qwen2.5-coder:32b -- python3 "$(pwd)/server.py"
```

<details>
<summary><b>Alternative: manual JSON config</b> (if you prefer editing the file directly)</summary>

Claude Code stores MCP server config in `~/.claude.json` (created automatically the first
time you add a server — it's a hidden file, so use `ls -la` to see it). You can add an
entry by hand instead of using the CLI:

```json
{
  "mcpServers": {
    "churn-analytics": {
      "command": "python3",
      "args": ["/absolute/path/to/CHEQ-Home-Assignment/server.py"]
    }
  }
}
```

> **Tip:** Replace `/absolute/path/to/CHEQ-Home-Assignment/server.py` with the real path
> on your machine. Run `pwd` inside the `CHEQ-Home-Assignment` directory to get it.

The exact config file/location can vary slightly between Claude Code versions and
platforms — if this doesn't work, `claude mcp add` (above) is the more reliable method
since it's handled internally by Claude Code itself.

</details>

---


## Example Questions

Once connected, try asking Claude Code:

```
What is the overall churn rate?
Which contract type has the highest churn?
What's the average monthly charge for churned vs retained customers?
How does churn rate vary by tenure length?
Which internet type (Fiber, DSL, Cable) has the highest churn?
What percentage of customers paying by credit card churn?
How much monthly revenue is lost to churn?
What are the top reasons customers cite for churning?
```

---

## Model Used

**Primary: [`qwen2.5-coder:14b`](https://ollama.com/library/qwen2.5-coder)** running
locally via [Ollama](https://ollama.com) — powers the `ask_question` tool, the intended
default way to query this server. Completely free, no API key, no account, no data
leaving your machine. Configure via:

```bash
export OLLAMA_MODEL="qwen2.5-coder:14b"          # default, no need to set explicitly
export OLLAMA_BASE_URL="http://localhost:11434"  # default, no need to set explicitly
```

Swap in any other Ollama model (`ollama pull <model>` then set `OLLAMA_MODEL`) — for
example `qwen2.5-coder:32b` for higher accuracy on machines with 48GB+ memory.

**Fallback: the LLM built into whichever MCP host you connect to** (Claude Code or
Codex). If Ollama isn't running, the other five tools (`get_schema`, `query_sql`,
`get_churn_summary`, etc.) still work — that host model can drive the SQL generation
directly instead. No configuration needed; this is free as part of your existing
Claude Code / Codex usage.

---

## Project Structure

```
CHEQ-Home-Assignment/
├── docs/
│   └── churn_mcp_design.pdf
├── server.py
├── setup_data.py
├── test_server.py
├── requirements.txt
├── README.md
└── .gitignore
└── data/
    └── churn.csv # Downloaded by setup_data.py
```
📄 **[One-Page Design Document (PDF)](./docs/churn_mcp_design.pdf)**

---

## Architecture Decision: SQL

The Customer Churn dataset is **structured tabular data**.
SQL gives exact, deterministic answers for aggregation questions like
"What % of fiber-optic customers churn?".

DuckDB requires no server, reads CSVs natively, and runs in-memory — perfect
for a local MCP server with no infrastructure overhead.

See the one-page design document for the full rationale.

---

## Security Notes

- The `query_sql` tool only accepts `SELECT` / `WITH` statements.
  `INSERT`, `UPDATE`, `DROP`, etc. are blocked.
- No data ever leaves the local machine. `ask_question` calls Ollama on `localhost` —
  there is no external network request, no API key, and no third-party data sharing.

---

