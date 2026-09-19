"""Text-to-SQL over the inventory database, with guardrails.

Security posture (this matters more than the prompt):

* the connection is opened **read-only** (SQLite ``mode=ro``);
* only a single ``SELECT``/``WITH`` statement is accepted – anything else
  (DDL, DML, PRAGMA, ATTACH, multiple statements) is rejected before execution;
* a ``LIMIT`` is enforced and a statement timeout guards against runaway queries;
* the LLM only ever sees the schema and the result rows, never credentials.

The LLM is asked for JSON ``{"sql": ..., "explanation": ...}`` so that the
generated statement can be validated and logged separately from the answer.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DB_PATH
from ..llm import LLM, extract_json

log = logging.getLogger(__name__)

SCHEMA_DESCRIPTION = """
Tables (SQLite):

projects(project_id TEXT PK, name_ar, name_en, city_ar, city_en, district_ar, district_en,
         launch_month 'YYYY-MM', delivery_month 'YYYY-MM', total_units INTEGER)

units(unit_id TEXT PK, project_id -> projects, unit_no, unit_type IN ('apartment','townhouse','duplex','villa'),
      unit_type_ar, bedrooms INTEGER, bathrooms INTEGER, area_sqm REAL, floor INTEGER (NULL for villas/townhouses),
      view IN ('garden','street','city','sea','park'), view_ar, parking_spaces INTEGER,
      list_price_sar REAL, status IN ('available','reserved','sold'), status_ar, delivery_month 'YYYY-MM')

sales(sale_id INTEGER PK, unit_id -> units, project_id -> projects, sale_date 'YYYY-MM-DD',
      sale_price_sar REAL, discount_pct REAL, payment_method IN ('bank_financing','cash'), bank_name, buyer_city)

Notes:
- Arabic project names are in projects.name_ar (e.g. 'واحة النرجس'), English in name_en (e.g. 'Narjis Oasis').
  Match names with LIKE '%...%' to be tolerant of partial names.
- Prices are in Saudi Riyals (SAR). "Available" means units.status = 'available'.
- Use strftime('%Y-%m', sale_date) for monthly aggregation.
""".strip()

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


class UnsafeSQL(ValueError):
    pass


def validate_sql(sql: str, *, max_rows: int = 50) -> str:
    """Return a safe, single, LIMITed SELECT statement or raise ``UnsafeSQL``."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise UnsafeSQL("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        raise UnsafeSQL("only SELECT statements are allowed")
    if _FORBIDDEN.search(s):
        raise UnsafeSQL("statement contains a forbidden keyword")
    if re.search(r"\blimit\b", s, re.IGNORECASE):
        # cap any user-provided limit
        s = re.sub(r"\blimit\s+(\d+)", lambda m: f"LIMIT {min(int(m.group(1)), max_rows)}", s, flags=re.IGNORECASE)
    else:
        s = f"{s} LIMIT {max_rows}"
    return s


@dataclass
class SQLResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    explanation: str = ""
    error: str | None = None
    attempts: int = 1

    def as_markdown(self, max_rows: int = 20) -> str:
        if self.error:
            return f"_query failed: {self.error}_"
        if not self.rows:
            return "_no rows_"
        head = "| " + " | ".join(self.columns) + " |\n|" + "---|" * len(self.columns)
        body = "\n".join("| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows[:max_rows])
        more = f"\n_… {len(self.rows) - max_rows} more rows_" if len(self.rows) > max_rows else ""
        return f"{head}\n{body}{more}"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"
    return "" if v is None else str(v)


class SQLAgent:
    def __init__(self, llm: LLM, db_path: Path = DB_PATH, max_rows: int = 50, timeout_s: float = 5.0):
        self.llm = llm
        self.db_path = Path(db_path)
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------ #
    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=self.timeout_s)
        # statement timeout via progress handler (~ every 10k VM steps)
        import time

        deadline = time.monotonic() + self.timeout_s
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        return con

    def execute(self, sql: str) -> SQLResult:
        safe = validate_sql(sql, max_rows=self.max_rows)
        con = self._connect()
        try:
            cur = con.execute(safe)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return SQLResult(sql=safe, columns=cols, rows=rows)
        finally:
            con.close()

    # ------------------------------------------------------------------ #
    def generate_sql(self, question: str, *, previous_error: str | None = None, previous_sql: str | None = None) -> tuple[str, str]:
        system = (
            "You translate questions about a real-estate developer's unit inventory into a single SQLite SELECT query.\n"
            "Rules: return ONLY a JSON object {\"sql\": \"...\", \"explanation\": \"...\"}; one SELECT statement; no comments; "
            "no data modification; prefer aggregates (COUNT, AVG, MIN, MAX) when the question asks 'how many'/'average'; "
            "include human-readable columns (project name, unit_no) rather than ids; add ORDER BY where useful.\n\n"
            + SCHEMA_DESCRIPTION
        )
        user = f"Question: {question}"
        if previous_error:
            user += f"\n\nYour previous SQL failed:\n{previous_sql}\nError: {previous_error}\nFix it."
        text = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=True)
        data = extract_json(text)
        if not isinstance(data, dict) or "sql" not in data:
            raise ValueError("LLM did not return {'sql': ...}")
        return str(data["sql"]), str(data.get("explanation", ""))

    def ask(self, question: str, *, max_attempts: int = 2) -> SQLResult:
        """Generate → validate → execute, with one repair round-trip on error."""
        sql, expl, err = None, "", None
        for attempt in range(1, max_attempts + 1):
            try:
                sql, expl = self.generate_sql(question, previous_error=err, previous_sql=sql)
                res = self.execute(sql)
                res.explanation = expl
                res.attempts = attempt
                return res
            except (UnsafeSQL, sqlite3.Error, ValueError) as e:
                err = str(e)
                log.warning("SQL attempt %d failed: %s", attempt, err)
        return SQLResult(sql=sql or "", columns=[], rows=[], explanation=expl, error=err, attempts=max_attempts)
