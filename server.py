#!/usr/bin/env python3
"""
Customer Churn Analytics — MCP Server
======================================
Exposes the Customer Churn dataset as a set of MCP tools for natural-language
Q&A, backed by SQL over an in-memory DuckDB database.

Tools
-----
ask_question(question)         – PRIMARY: ask any question in plain English; the
                                  server's own local LLM (Ollama) turns it into SQL,
                                  runs it, and returns the answer
get_schema()                   – column names + types (building block / power-user)
get_sample_rows(n)             – first N rows (building block / power-user)
query_sql(sql)                 – run any SELECT / WITH query directly (power-user;
                                  use when ask_question's SQL needs manual tweaking)
get_churn_summary()            – pre-computed overview stats
get_column_distribution(col)   – value counts or numeric stats for one column
"""

import os
import re
import json
import urllib.request
import urllib.error
import duckdb
import pandas as pd
from mcp.server.fastmcp import FastMCP

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "data", "churn.csv")

# ── App ───────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    "churn-analytics",
    instructions=(
        "This server answers natural-language questions about the Customer Churn dataset. "
        "For any user question about churn rates, customer segments, revenue impact, or risk "
        "factors, call ask_question() FIRST — it runs a dedicated local LLM (Ollama) to convert "
        "the question into SQL and returns the answer directly. Only fall back to query_sql(), "
        "get_schema(), or the other tools if ask_question() is unavailable (e.g. Ollama isn't "
        "running) or its generated SQL needs manual correction."
    ),
)

# ── Database (lazy-init) ──────────────────────────────────────────────────────
_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Dataset not found: {DATA_PATH}\n"
                "Run first:  python setup_data.py"
            )
        _conn = duckdb.connect(":memory:")
        _conn.execute(
            f"CREATE TABLE churn AS SELECT * FROM read_csv_auto('{DATA_PATH}', header=true)"
        )
    return _conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_select(sql: str) -> str:
    """Reject anything that isn't a read-only statement."""
    clean = sql.strip().lstrip(";").strip().upper()
    if not (clean.startswith("SELECT") or clean.startswith("WITH")):
        raise ValueError("Only SELECT / WITH queries are allowed.")
    return sql


def _df_to_str(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "(no rows returned)"
    if len(df) > max_rows:
        snippet = df.head(max_rows).to_string(index=False)
        return f"{snippet}\n\n… {len(df) - max_rows} additional rows truncated."
    return df.to_string(index=False)


def _schema_df() -> pd.DataFrame:
    return get_conn().execute("DESCRIBE churn").fetchdf()


def _normalize(name: str) -> str:
    """Lowercase and strip everything but letters/digits, so 'Monthly Charge' == 'monthlycharges'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_column(*candidates: str) -> str | None:
    """Find a real column name matching any of the normalized candidate names."""
    schema = _schema_df()
    norm_map = {_normalize(c): c for c in schema["column_name"]}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    return None


def _churn_col_info() -> tuple[str, str]:
    """
    Return (column_name, churn_expr) for the churn target column.

    In this dataset (aai510-group1/telco-customer-churn) the target is a single
    "Churn" column, already encoded as 0/1 (BIGINT). This function still checks
    the actual values rather than assuming that encoding, so it degrades gracefully
    if pointed at a dataset variant where churn is stored as 'Yes'/'No' text instead.
    """
    churn_col = _find_column("churn")
    if not churn_col:
        raise ValueError("No 'Churn' column found in dataset.")

    sample = get_conn().execute(
        f'SELECT DISTINCT "{churn_col}" FROM churn LIMIT 5'
    ).fetchdf().iloc[:, 0].astype(str).tolist()
    if any(v in ("Yes", "No", "TRUE", "FALSE") for v in sample):
        expr = f'CASE WHEN "{churn_col}" IN (\'Yes\', \'True\', \'1\') THEN 1 ELSE 0 END'
    else:
        expr = f'CAST("{churn_col}" AS INTEGER)'
    return churn_col, expr


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def ask_question(question: str) -> str:
    """
    [PRIMARY TOOL] Ask any natural-language question about the churn data and get a
    direct answer — start here for any user question about churn rates, customer
    segments, revenue impact, or risk factors.

    This tool runs a dedicated local LLM (Ollama, qwen2.5-coder by default) that
    converts your question into SQL, executes it, and returns the result. It requires
    no schema lookup or manual SQL writing on your part — pass the question as-is.

    Requires Ollama running locally (https://ollama.com) with a model pulled — see
    README. If Ollama isn't reachable, this tool returns clear setup instructions
    instead of failing silently; in that case, fall back to query_sql() or
    get_churn_summary() to still answer the question.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")

    schema_str = get_schema()
    sample_str = get_sample_rows(3)

    system_prompt = (
        "You are a DuckDB SQL expert. Convert the user's question into a single DuckDB SQL query.\n"
        f"Table name: churn\n\nSchema:\n{schema_str}\n\nSample rows:\n{sample_str}\n\n"
        "Rules:\n"
        "- Output ONLY the SQL query — no explanation, no markdown fences, no comments.\n"
        "- Use only SELECT / WITH statements.\n"
        "- Column names containing spaces MUST be double-quoted, e.g. \"Monthly Charge\".\n"
        "- Boolean-like columns (yes/no questions, e.g. churn, partner, dependents, "
        "phone service) may be encoded as BIGINT 0/1 OR as VARCHAR 'Yes'/'No' depending on "
        "the dataset — always check the column's type and the sample rows above to use the "
        "correct encoding. Do not assume 'Yes'/'No' if the type is BIGINT/INTEGER.\n"
        "- Prefer ROUND() for percentages (2 decimal places).\n"
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return (
            f"Could not reach Ollama at {base_url} ({exc}).\n\n"
            "Setup steps:\n"
            "  1. Install Ollama: https://ollama.com\n"
            f"  2. Pull a model:   ollama pull {model}\n"
            "  3. Ollama runs automatically on localhost:11434 after install.\n\n"
            "Or just use query_sql() directly with your own SQL — no LLM required for that tool."
        )
    except Exception as exc:
        return f"Ollama request failed: {exc}"

    sql = body.get("message", {}).get("content", "").strip()
    if not sql:
        return f"Ollama returned an empty response. Raw payload: {body}"

    # Strip accidental markdown fences
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()

    result = query_sql(sql)
    return f"Generated SQL:\n{sql}\n\nResult:\n{result}"


@mcp.tool()
def get_schema() -> str:
    """
    [Power-user tool] Return the column names and data types of the churn table.
    For answering a user's natural-language question, prefer ask_question() instead —
    it handles schema lookup internally. Use this directly only if you're writing
    raw SQL by hand via query_sql(), or ask_question() is unavailable.
    """
    df = _schema_df()[["column_name", "column_type"]]
    return df.to_string(index=False)


@mcp.tool()
def get_sample_rows(n: int = 5) -> str:
    """
    [Power-user tool] Return the first N rows of the churn dataset (max 50).
    Useful for understanding actual values, formats, and encoding (e.g. 'Yes'/'No' vs 1/0)
    when writing raw SQL via query_sql(). For answering a user's question directly,
    prefer ask_question() instead.
    """
    n = max(1, min(n, 50))
    df = get_conn().execute(f"SELECT * FROM churn LIMIT {n}").fetchdf()
    return _df_to_str(df)


@mcp.tool()
def query_sql(sql: str) -> str:
    """
    [Power-user tool] Execute a read-only SQL query against the churn table directly
    (DuckDB dialect). Only SELECT and WITH statements are permitted.

    Prefer ask_question() for answering a user's natural-language question — it
    generates and runs the SQL for you. Use query_sql() directly only when:
    (a) ask_question()'s generated SQL needs manual correction, (b) Ollama isn't
    available, or (c) you need a precise query ask_question() can't be trusted
    to construct correctly (e.g. complex multi-step analysis).

    Table name : churn
    DuckDB extras: QUALIFY, PIVOT, MEDIAN(), PERCENTILE_CONT(), regexp_matches(), etc.
    Note: many column names contain spaces (e.g. "Monthly Charge", "Tenure in Months",
    "Payment Method") — wrap them in double quotes.
    Note: the target column is "Churn" and it is already 0/1 (BIGINT), not 'Yes'/'No'.
    Most other boolean-like columns (Partner, Dependents, Senior Citizen, Phone Service,
    Paperless Billing, Online Security, Streaming TV, etc.) are also 0/1 BIGINT, not
    'Yes'/'No' strings — always check get_sample_rows() if unsure how a column is encoded.

    Examples
    --------
    SELECT COUNT(*) FROM churn WHERE "Churn" = 1
    SELECT Contract, ROUND(AVG("Monthly Charge"), 2) AS avg_charge
      FROM churn GROUP BY Contract ORDER BY avg_charge DESC
    SELECT tenure_bucket, churn_rate FROM (
        SELECT CASE WHEN "Tenure in Months" < 12 THEN '0-12m'
                    WHEN "Tenure in Months" < 24 THEN '12-24m'
                    ELSE '24m+' END AS tenure_bucket,
               ROUND(100.0 * AVG("Churn"), 2) AS churn_rate
        FROM churn GROUP BY 1
    ) ORDER BY churn_rate DESC
    """
    conn = get_conn()
    try:
        sql = _safe_select(sql)
        df = conn.execute(sql).fetchdf()
        return _df_to_str(df)
    except ValueError as exc:
        return f"Permission denied: {exc}"
    except duckdb.Error as exc:
        return f"SQL error: {exc}"


@mcp.tool()
def get_churn_summary() -> str:
    """
    Return a pre-computed churn overview:
    overall rate, breakdown by contract type, internet service, payment method,
    and average monthly/total charges for churned vs retained customers.

    Useful as a fast shortcut for broad "give me an overview" questions, or as a
    fallback if ask_question() is unavailable. For a specific natural-language
    question, prefer ask_question() — it's more precise for narrow questions
    than this fixed summary.
    """
    conn = get_conn()

    try:
        churn_col, churn_expr = _churn_col_info()
    except ValueError as e:
        return str(e)

    sections: list[str] = []

    # ── Overall ──────────────────────────────────────────────────────────────
    df = conn.execute(f"""
        SELECT
            COUNT(*)                                              AS total_customers,
            SUM({churn_expr})                                    AS churned,
            COUNT(*) - SUM({churn_expr})                         AS retained,
            ROUND(100.0 * SUM({churn_expr}) / COUNT(*), 2)       AS churn_rate_pct
        FROM churn
    """).fetchdf()
    sections.append("### Overall\n" + df.to_string(index=False))

    # ── By Contract ──────────────────────────────────────────────────────────
    col = _find_column("contract", "contracttype")
    if col:
        df = conn.execute(f"""
            SELECT "{col}"                                              AS contract,
                   COUNT(*)                                             AS customers,
                   ROUND(100.0 * SUM({churn_expr}) / COUNT(*), 2)      AS churn_rate_pct
            FROM churn GROUP BY "{col}" ORDER BY churn_rate_pct DESC
        """).fetchdf()
        sections.append("### By Contract Type\n" + df.to_string(index=False))

    # ── By Internet Type ───────────────────────────────────────────────────────
    col = _find_column("internettype", "internetservice")
    if col:
        df = conn.execute(f"""
            SELECT "{col}"                                              AS internet_type,
                   COUNT(*)                                             AS customers,
                   ROUND(100.0 * SUM({churn_expr}) / COUNT(*), 2)      AS churn_rate_pct
            FROM churn GROUP BY "{col}" ORDER BY churn_rate_pct DESC
        """).fetchdf()
        sections.append("### By Internet Type\n" + df.to_string(index=False))

    # ── By Payment Method ─────────────────────────────────────────────────────
    col = _find_column("paymentmethod")
    if col:
        df = conn.execute(f"""
            SELECT "{col}"                                              AS payment_method,
                   COUNT(*)                                             AS customers,
                   ROUND(100.0 * SUM({churn_expr}) / COUNT(*), 2)      AS churn_rate_pct
            FROM churn GROUP BY "{col}" ORDER BY churn_rate_pct DESC
        """).fetchdf()
        sections.append("### By Payment Method\n" + df.to_string(index=False))

    # ── Revenue Impact ────────────────────────────────────────────────────────
    mc_col = _find_column("monthlycharges", "monthlycharge")
    if mc_col:
        df = conn.execute(f"""
            SELECT
                CASE WHEN {churn_expr} = 1 THEN 'Churned' ELSE 'Retained' END AS segment,
                COUNT(*)                                                AS customers,
                ROUND(AVG("{mc_col}"), 2)                               AS avg_monthly_charges,
                ROUND(SUM("{mc_col}"), 2)                               AS total_monthly_revenue
            FROM churn GROUP BY 1 ORDER BY 1
        """).fetchdf()
        sections.append("### Revenue by Segment\n" + df.to_string(index=False))

    return "\n\n".join(sections)


@mcp.tool()
def get_column_distribution(column: str) -> str:
    """
    Return the value distribution for a given column.
    • Categorical columns → count and percentage per unique value (top 30).
    • Numeric columns     → min, max, mean, median, stddev, percentiles.
    Pass the exact column name (case-insensitive).

    Useful for quick single-column exploration. For a specific natural-language
    question, prefer ask_question() instead.
    """
    conn = get_conn()
    schema = _schema_df()
    col_names = schema["column_name"].tolist()
    col_types = schema["column_type"].tolist()

    match = next((c for c in col_names if c.lower() == column.lower()), None)
    if match is None:
        return f"Column '{column}' not found. Available: {', '.join(col_names)}"

    idx = col_names.index(match)
    col_type = col_types[idx].upper()
    is_numeric = any(
        t in col_type for t in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "BIGINT", "HUGEINT", "REAL")
    )

    if is_numeric:
        df = conn.execute(f"""
            SELECT
                COUNT(*)                                 AS count,
                COUNT(*) - COUNT("{match}")              AS nulls,
                ROUND(MIN("{match}"), 4)                 AS min,
                ROUND(MAX("{match}"), 4)                 AS max,
                ROUND(AVG("{match}"), 4)                 AS mean,
                ROUND(MEDIAN("{match}"), 4)              AS median,
                ROUND(STDDEV("{match}"), 4)              AS stddev,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{match}"), 4) AS p25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{match}"), 4) AS p75
            FROM churn
        """).fetchdf()
    else:
        total = conn.execute("SELECT COUNT(*) FROM churn").fetchone()[0]
        df = conn.execute(f"""
            SELECT
                "{match}"                              AS value,
                COUNT(*)                               AS count,
                ROUND(100.0 * COUNT(*) / {total}, 1)  AS pct
            FROM churn
            GROUP BY "{match}"
            ORDER BY count DESC
            LIMIT 30
        """).fetchdf()

    header = f"Column: {match}  |  Type: {col_types[idx]}\n"
    return header + df.to_string(index=False)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")
