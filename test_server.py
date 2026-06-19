#!/usr/bin/env python3
"""
test_server.py
===============
Standalone verification script — exercises every tool in server.py end-to-end,
including ask_question (which requires a running local Ollama instance).

Usage
-----
    python test_server.py

This imports server.py directly and calls each tool function (no MCP protocol
overhead needed for this kind of check — for a full protocol-level test, use
Claude Code's own verification, e.g. its `/run` feature, which drives the
server over stdio as a real MCP client would).

Exit code is 0 if all required checks pass, 1 otherwise.
ask_question is checked but never fails the suite — if Ollama isn't running,
it's reported as SKIPPED with setup instructions, not FAILED, since it's an
optional bonus tool.
"""

import os
import sys
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []  # (test name, status, detail)


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "—"}[status]
    print(f"{icon} [{status}] {name}" + (f"  —  {detail}" if detail else ""))


def main() -> int:
    print("=== Customer Churn MCP Server — Verification ===\n")

    # ── 0. Ensure dataset exists ────────────────────────────────────────────
    data_path = os.path.join(HERE, "data", "churn.csv")
    if not os.path.exists(data_path):
        print("data/churn.csv not found — running setup_data.py first...\n")
        import setup_data
        ok = setup_data.try_huggingface()
        if not ok:
            record("dataset download", FAIL, "setup_data.py failed — see error above")
            print_summary()
            return 1
        print()

    try:
        import server
    except Exception as exc:
        record("import server.py", FAIL, str(exc))
        print_summary()
        return 1
    record("import server.py", PASS)

    # ── 1. get_schema ────────────────────────────────────────────────────────
    try:
        schema = server.get_schema()
        assert "Churn" in schema, "expected 'Churn' column not found in schema"
        record("get_schema()", PASS, f"{schema.count(chr(10))} columns found")
    except Exception as exc:
        record("get_schema()", FAIL, str(exc))

    # ── 2. get_sample_rows ──────────────────────────────────────────────────
    try:
        sample = server.get_sample_rows(3)
        assert "Customer ID" in sample or len(sample) > 0
        record("get_sample_rows(3)", PASS)
    except Exception as exc:
        record("get_sample_rows(3)", FAIL, str(exc))

    # ── 3. query_sql (valid query) ──────────────────────────────────────────
    try:
        result = server.query_sql('SELECT COUNT(*) AS n FROM churn')
        assert "n" in result
        record("query_sql() — valid SELECT", PASS, result.strip().replace(chr(10), " | "))
    except Exception as exc:
        record("query_sql() — valid SELECT", FAIL, str(exc))

    # ── 4. query_sql (security guard) ───────────────────────────────────────
    try:
        blocked = server.query_sql("DROP TABLE churn")
        assert "Permission denied" in blocked
        record("query_sql() — blocks non-SELECT", PASS)
    except Exception as exc:
        record("query_sql() — blocks non-SELECT", FAIL, str(exc))

    # ── 5. get_churn_summary ────────────────────────────────────────────────
    try:
        summary = server.get_churn_summary()
        assert "Overall" in summary
        record("get_churn_summary()", PASS)
    except Exception as exc:
        record("get_churn_summary()", FAIL, str(exc))

    # ── 6. get_column_distribution ──────────────────────────────────────────
    try:
        dist = server.get_column_distribution("Contract")
        assert "Column: Contract" in dist or "Contract" in dist
        record("get_column_distribution('Contract')", PASS)
    except Exception as exc:
        record("get_column_distribution('Contract')", FAIL, str(exc))

    # ── 7. ask_question (requires Ollama — skipped gracefully if unavailable) ─
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_up = False
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3) as resp:
            ollama_up = resp.status == 200
    except Exception:
        ollama_up = False

    if not ollama_up:
        record(
            "ask_question() — Ollama check",
            SKIP,
            f"Ollama not reachable at {ollama_url}. Run `ollama serve` and "
            f"`ollama pull qwen2.5-coder:14b`, then re-run this script.",
        )
    else:
        try:
            answer = server.ask_question("What is the overall churn rate?")
            if "Generated SQL" in answer and "SQL error" not in answer:
                record("ask_question() — NL to SQL via Ollama", PASS, answer.splitlines()[1][:80])
            else:
                record("ask_question() — NL to SQL via Ollama", FAIL, answer[:200])
        except Exception as exc:
            record("ask_question() — NL to SQL via Ollama", FAIL, str(exc))

    return print_summary()


def print_summary() -> int:
    print("\n=== Summary ===")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped (of {len(results)} checks)")
    if n_fail > 0:
        print("\n❌ Some required checks failed.")
        return 1
    if n_skip > 0:
        print("\n✓ All required checks passed (ask_question skipped — Ollama not running).")
    else:
        print("\n✓ All checks passed, including ask_question via Ollama.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
