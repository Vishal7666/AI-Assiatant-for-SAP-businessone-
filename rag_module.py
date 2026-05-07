"""
rag_module.py — SAP B1 RAG — Mahesh Industries
─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE:
  • Groq  → ONLY for NL→SQL + DB query interpretation (what it's best at)
  • Gemini → ONLY for general chat, weather, jokes, current events, etc.
  • Smart router: keyword + LLM-based decision to pick the right engine

FLOW:
  User Question
    │
    ▼
  [Router] — is this a DB question?
    │
    ├─ YES → Groq generates T-SQL → MSSQL executes → format answer
    │         (auto-fix on SQL error, retry up to 3 times)
    │
    └─ NO  → Gemini answers freely (weather, jokes, general knowledge, etc.)
─────────────────────────────────────────────────────────────────────────────
"""

import re
import time
from decimal import Decimal
from datetime import date, datetime

import pyodbc
from groq import Groq, RateLimitError


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class MSSQLConfig:
    SERVER   = r"localhost\SQLEXPRESS"
    PORT     = 1433
    USER     = "sa"
    PASSWORD = "Admin@1234"
    DRIVER   = "ODBC Driver 17 for SQL Server"
    DATABASE = "Mahesh_Industries"
    MAX_ROWS = 3000


# ══════════════════════════════════════════════════════════════════════════════
#  COMPACT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

SAP_SCHEMA = """
SAP Business One — Mahesh_Industries (MS SQL Server). No table prefix needed.

-- Business Partners
OCRD  (CardCode PK, CardName, CardType[C=Customer,S=Vendor/Supplier],
        E_Mail, Phone1, Cellular, City, Balance, CreditLine,
        validFor[Y/N], CreateDate)
OCPR  (CardCode FK, Name, E_MailL, Tel1, Cellolar, Position)  -- Contact persons

-- Sales Team
OSLP  (SlpCode PK, SlpName, Email, Telephone, Mobil, Active)

-- Sales Documents
OINV  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocDueDate,
        DocTotal, PaidToDate, VatSum, VatPercent, DocStatus[O=Open,C=Closed], SlpCode)
INV1  (DocEntry FK, ItemCode, Dscription, Quantity, Price, LineTotal, WhsCode)

ORDR  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocDueDate,
        DocTotal, DocStatus[O/C], SlpCode)
RDR1  (DocEntry FK, ItemCode, Dscription, Quantity, Price, LineTotal)

OQUT  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocDueDate, DocTotal, DocStatus[O/C])
ODLN  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocTotal, DocStatus[O/C])  -- Deliveries
ORIN  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocTotal, DocStatus[O/C])  -- Sales Credit Notes

-- Purchase Documents
OPOR  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocDueDate, DocTotal, DocStatus[O/C])
POR1  (DocEntry FK, ItemCode, Dscription, Quantity, Price, LineTotal)

OPCH  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocTotal,
        PaidToDate, VatSum, DocStatus[O/C])
PCH1  (DocEntry FK, ItemCode, Dscription, Quantity, Price, LineTotal)

-- Payments
OINC  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocTotal)
OVPM  (DocEntry PK, DocNum, CardCode, CardName, DocDate, DocTotal)

-- Inventory
OITM  (ItemCode PK, ItemName, OnHand, IsCommited, OnOrder, MinLevel,
        AvgPrice, validFor[Y/N], InvntItem[Y/N], ItmsGrpCod, DfltWH)
OITG  (ItmsGrpCod PK, ItmsGrpNam)
OITW  (ItemCode FK, WhsCode FK, OnHand, IsCommited, OnOrder)
OWHS  (WhsCode PK, WhsName, City)

-- Production
OWOR  (DocEntry PK, DocNum, ItemCode, ProdName, Status[P=Planned,R=Released,L=Completed],
        PlannedQty, CmpltQty, RjctQty, DueDate, CloseDate, Warehouse)
WOR1  (DocEntry FK, ItemCode, ItemName, PlannedQty, IssuedQty, wareHouse)

-- GL / Accounts
OACT  (AcctCode PK, AcctName, CurrTotal, ActType)

-- KEY JOINS --
OCRD.CardCode = OINV.CardCode = ORDR.CardCode = OPCH.CardCode = OPOR.CardCode
OINV.DocEntry = INV1.DocEntry
ORDR.DocEntry = RDR1.DocEntry
OPCH.DocEntry = PCH1.DocEntry
OITM.ItemCode = OITW.ItemCode = INV1.ItemCode = PCH1.ItemCode
OCRD.CardCode = OCPR.CardCode (one-to-many)
OSLP.SlpCode  = OINV.SlpCode = ORDR.SlpCode

-- COLUMN NOTES --
• Contact email  : OCPR.E_MailL  (not E_Mail)
• Contact mobile : OCPR.Cellolar (not Cellular)
• Item desc line : INV1.Dscription (not Description)
• Tax column     : OINV.VatSum / OPCH.VatSum
• Customer mobile: OCRD.Cellular
• Customer email : OCRD.E_Mail
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SQL_SYSTEM = """You are a T-SQL expert for SAP Business One (Mahesh Industries MS SQL Server).

STRICT RULES:
- Return ONLY a raw T-SQL SELECT statement. Nothing else.
- NO markdown, NO backticks, NO explanation, NO comments.
- Always add TOP {max_rows} unless the query uses COUNT/SUM/AVG/MIN/MAX aggregates.
- Use GETDATE() for current date. Use DATEADD/DATEDIFF for date math.
- Use exact column names from the schema below. Always use table aliases in JOINs.
- Never prefix tables with database/schema names like [dbo]. or Mahesh_Industries.dbo.
- For partial name searches use: CardName LIKE '%keyword%'
- For current year: YEAR(DocDate) = YEAR(GETDATE())
- For current month: YEAR(DocDate)=YEAR(GETDATE()) AND MONTH(DocDate)=MONTH(GETDATE())
- If the question is NOT about business/company data, return exactly the word: GENERAL_CHAT"""

SQL_USER = """Schema:
{schema}

Question: {question}

T-SQL (SELECT only, no markdown):"""

FIX_SYSTEM = """You are a T-SQL expert. Fix the broken SQL query.
Return ONLY the corrected SELECT statement. No markdown, no explanation, no backticks."""

FIX_USER = """Schema (relevant part):
{schema}

Original question: {question}
Broken SQL: {sql}
Error message: {error}

Fixed T-SQL:"""

ANSWER_SYSTEM = """You are a helpful business assistant for Mahesh Industries (SAP B1).
Given database query results, provide a clear concise answer in 1-3 sentences.
Use ₹ for Indian Rupee amounts. Use Indian number formatting (lakhs/crores where appropriate)."""

ANSWER_USER = """Question: {question}
Database result: {data}
Answer:"""


# ══════════════════════════════════════════════════════════════════════════════
#  KEYWORD ROUTER — fast pre-check before spending a Groq call
# ══════════════════════════════════════════════════════════════════════════════

# Strong DB signals — if ANY of these match → definitely a DB question
_DB_KEYWORDS = {
    # documents
    "invoice", "invoices", "order", "orders", "quotation", "quotations",
    "purchase", "delivery", "deliveries", "payment", "payments", "receipt",
    "credit note", "grn",
    # business entities
    "customer", "customers", "vendor", "vendors", "supplier", "suppliers",
    "partner", "partners", "contact", "salesperson", "sales rep",
    # finance
    "revenue", "sales", "balance", "outstanding", "paid", "unpaid",
    "receivable", "payable", "gst", "vat", "tax", "profit", "loss",
    "amount", "total", "sum",
    # inventory
    "stock", "inventory", "item", "items", "product", "products",
    "warehouse", "quantity", "reorder", "minimum level",
    # production
    "production", "work order", "manufacture", "manufactured",
    # time-scoped business
    "this month", "last month", "this year", "last year", "today",
    "this week", "quarter", "ytd",
    # sap specific
    "docnum", "cardcode", "cardname", "itemcode",
}

# Strong NON-DB signals — if ANY of these match → definitely general chat
_GENERAL_KEYWORDS = {
    "weather", "temperature", "rain", "forecast", "climate",
    "joke", "jokes", "funny", "laugh",
    "news", "headline", "headlines",
    "hello", "hi ", " hi", "hey", "how are you", "good morning",
    "good afternoon", "good evening", "what is your name",
    "capital of", "population of", "president", "prime minister",
    "cricket", "football", "ipl", "match", "score",
    "recipe", "cook", "movie", "film", "song",
    "translate", "meaning of", "define ", "definition",
    "math", "calculate", "2+2", "what is 2",
}


def _route_question(question: str) -> str:
    """
    Returns 'DB' or 'GENERAL' based on keyword pre-screening.
    Returns 'UNSURE' if ambiguous (needs Groq to decide).
    """
    q_lower = question.lower()

    # Check general keywords first (faster exit)
    for kw in _GENERAL_KEYWORDS:
        if kw in q_lower:
            return "GENERAL"

    # Check DB keywords
    for kw in _DB_KEYWORDS:
        if kw in q_lower:
            return "DB"

    return "UNSURE"


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CLIENT — robust with retries and model fallback
# ══════════════════════════════════════════════════════════════════════════════

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


def _make_groq_client(api_key: str) -> tuple:
    client = Groq(api_key=api_key)
    for model in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            _ = resp.choices[0].message.content
            print(f"  ✅ [Groq] Model: {model}")
            return client, model
        except Exception as e:
            print(f"  ❌ [Groq] {model}: {str(e)[:60]}")
            continue
    print("  ⚠️  [Groq] Falling back to llama3-8b-8192")
    return client, "llama3-8b-8192"


def _call_groq(
    client: Groq,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 600,
    temperature: float = 0.0,
    max_retries: int = 4,
) -> str:
    """Call Groq with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = (resp.choices[0].message.content or "").strip()
            return result
        except RateLimitError:
            wait = 3 * (2 ** attempt)   # 3, 6, 12, 24
            print(f"  ⏳ [Groq] Rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
        except Exception as e:
            err = str(e)
            print(f"  ❌ [Groq] Error attempt {attempt+1}: {err[:80]}")
            if attempt < max_retries - 1:
                time.sleep(2)
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SQL EXECUTOR — safe, clean, robust
# ══════════════════════════════════════════════════════════════════════════════

class _SQLRunner:
    def __init__(self, cfg: MSSQLConfig):
        self.cfg = cfg
        self._cs = (
            f"DRIVER={{{cfg.DRIVER}}};"
            f"SERVER={cfg.SERVER},{cfg.PORT};"
            f"UID={cfg.USER};PWD={cfg.PASSWORD};"
            f"DATABASE={cfg.DATABASE};"
            f"TrustServerCertificate=yes;"
            f"Connection Timeout=15;"
        )
        self._verify()

    def _connect(self):
        return pyodbc.connect(self._cs, autocommit=True)

    def _verify(self):
        try:
            self._connect().close()
            print(f"  ✅ [SQL] Connected to [{self.cfg.DATABASE}]")
        except Exception as e:
            raise ConnectionError(f"❌ [SQL] Connection failed: {e}")

    def _clean_sql(self, raw: str) -> str:
        """Strip markdown, schema prefixes, leading semicolons."""
        sql = re.sub(r"```(?:sql|tsql|SQL)?", "", raw)
        sql = re.sub(r"```", "", sql).strip()
        sql = sql.lstrip(";").strip()

        # Remove any database/schema prefix patterns
        prefixes = [
            r"\[Mahesh_Industries\]\.\[dbo\]\.",
            r"Mahesh_Industries\.dbo\.",
            r"\[dbo\]\.",
            r"dbo\.",
            r"\[Mahesh_Industries\]\.",
            r"Mahesh_Industries\.",
        ]
        for pat in prefixes:
            sql = re.sub(pat, "", sql, flags=re.IGNORECASE)

        return sql.strip()

    def run(self, raw_sql: str) -> tuple:
        """Execute SQL. Returns (rows: list[dict], error: str)."""
        sql = self._clean_sql(raw_sql)

        # Safety: only SELECT allowed
        if not re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
            return [], f"Only SELECT statements are allowed. Got: {sql[:80]}"

        # Add TOP if missing and no aggregates present
        has_aggregate = re.search(
            r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE
        )
        has_top = re.search(r"\bTOP\b", sql, re.IGNORECASE)
        has_fetch = re.search(r"\bFETCH\b", sql, re.IGNORECASE)

        if not has_aggregate and not has_top and not has_fetch:
            sql = re.sub(
                r"^\s*SELECT\b",
                f"SELECT TOP {self.cfg.MAX_ROWS}",
                sql, count=1, flags=re.IGNORECASE,
            )

        print(f"  [SQL] Executing: {sql[:120]}...")

        try:
            conn = self._connect()
            cur  = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = []
            for row in cur.fetchall():
                record = {}
                for col, val in zip(cols, row):
                    if isinstance(val, Decimal):
                        val = float(val)
                    record[col] = val
                rows.append(record)
            cur.close()
            conn.close()
            print(f"  [SQL] ✅ {len(rows)} rows returned")
            return rows, ""
        except Exception as e:
            err = str(e)
            print(f"  [SQL] ❌ Error: {err[:120]}")
            return [], err


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATTER — clean HTML output
# ══════════════════════════════════════════════════════════════════════════════

_MONEY_WORDS = {
    "total", "amount", "revenue", "sales", "balance", "paid", "outstanding",
    "purchase", "income", "price", "cost", "profit", "credit", "debit",
    "value", "vat", "tax", "gst", "sum", "avg", "average", "due",
    "received", "lineTotal", "docTotal",
}

def _is_money(col: str) -> bool:
    return any(w in col.lower() for w in _MONEY_WORDS)

def _fmt_val(col: str, val) -> str:
    if val is None:
        return "—"
    if isinstance(val, (date, datetime)):
        return val.strftime("%d-%m-%Y")
    try:
        f = float(val)
        if _is_money(col):
            return f"₹{f:,.2f}"
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        s = str(val)
        return s[:50] if len(s) > 50 else s

def _pretty_col(col: str) -> str:
    s = re.sub(r"([A-Z])", r" \1", col).replace("_", " ").strip()
    return s.title()

def _html_table(rows: list) -> str:
    if not rows:
        return "<p>No records found.</p>"
    headers = list(rows[0].keys())[:8]
    th = "".join(
        f'<th style="padding:10px 14px;text-align:left;border-bottom:2px solid #383838;'
        f'white-space:nowrap;font-size:12px;color:#8e8ea0;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.5px;">{_pretty_col(c)}</th>'
        for c in headers
    )
    trs = []
    for i, row in enumerate(rows[:2000]):
        bg = "transparent" if i % 2 == 0 else "#252525"
        tds = ""
        for c in headers:
            v   = _fmt_val(c, row.get(c))
            aln = "right" if _is_money(c) else "left"
            wt  = "600" if c == headers[0] else "400"
            tds += (
                f'<td style="padding:10px 14px;border-bottom:1px solid #2a2a2a;'
                f'text-align:{aln};font-weight:{wt};font-size:13px;'
                f'white-space:nowrap;color:#ececec;">{v}</td>'
            )
        trs.append(f'<tr style="background:{bg};">{tds}</tr>')

    shown    = min(len(rows), 2000)
    footer   = (
        f"{shown} of {len(rows)} records shown"
        if len(rows) > 2000 else
        f"{len(rows)} record{'s' if len(rows) != 1 else ''}"
    )
    return (
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #383838;margin:14px 0;">'
        f'<table style="border-collapse:collapse;width:100%;min-width:400px;background:#1e1e1e;">'
        f'<thead style="background:#161616;"><tr>{th}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody>'
        f'</table>'
        f'<div style="padding:8px 14px;font-size:11px;color:#8e8ea0;border-top:1px solid #383838;'
        f'background:#161616;text-align:right;">'
        f'{footer} — Mahesh Industries SAP B1'
        f'</div></div>'
    )

def _single_card(row: dict) -> str:
    lines = ""
    for k, v in row.items():
        if v is not None:
            lines += (
                f'<tr><td style="padding:10px 14px;font-size:12px;color:#8e8ea0;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.4px;white-space:nowrap;'
                f'width:40%;border-bottom:1px solid #2a2a2a;">{_pretty_col(k)}</td>'
                f'<td style="padding:10px 14px;font-size:13.5px;color:#ececec;'
                f'border-bottom:1px solid #2a2a2a;">{_fmt_val(k, v)}</td></tr>'
            )
    return (
        f'<div style="overflow-x:auto;border-radius:10px;border:1px solid #383838;margin:14px 0;">'
        f'<table style="border-collapse:collapse;width:100%;background:#1e1e1e;">'
        f'<tbody>{lines}</tbody>'
        f'</table>'
        f'<div style="padding:8px 14px;font-size:11px;color:#8e8ea0;border-top:1px solid #383838;'
        f'background:#161616;text-align:right;">'
        f'1 record — Mahesh Industries SAP B1'
        f'</div></div>'
    )

def _format_table(rows: list) -> str:
    if not rows:
        return "<p>No records found.</p>"
    return _single_card(rows[0]) if len(rows) == 1 else _html_table(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE RAG MODULE
# ══════════════════════════════════════════════════════════════════════════════

class RAGModule:
    """
    Smart router:
      DB questions  → Groq (NL→SQL) + MSSQL execution
      General chat  → Gemini (passed in as gemini_module)

    Usage:
      rag = RAGModule(api_key=GROQ_KEY)
      answer, sources = rag.ask("total revenue this month", gemini_module=gemini)
      answer, sources = rag.ask("what is the weather in pune", gemini_module=gemini)
    """

    def __init__(self, cfg: MSSQLConfig = None, api_key: str = None):
        self.cfg = cfg or MSSQLConfig()
        print("\n" + "=" * 60)
        print("🚀  RAG Module — Mahesh Industries SAP B1")
        print("    DB questions → Groq+SQL  |  General → Gemini")
        print("=" * 60)

        self._gc = None
        self._gm = None

        if api_key:
            try:
                self._gc, self._gm = _make_groq_client(api_key)
            except Exception as e:
                print(f"  ⚠️  Groq init failed: {e}")

        try:
            self._sql = _SQLRunner(self.cfg)
        except ConnectionError as e:
            print(e)
            self._sql = None

        print("✅  RAG Ready\n" + "=" * 60)

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, user_message: str, gemini_module=None, ai_module=None, allow_db: bool = True) -> tuple:
        """
        Main entry point.
        gemini_module : GeminiModule instance for general questions
        ai_module     : legacy parameter (ignored — use gemini_module)
        allow_db      : if False, strictly block DB queries
        """
        q = user_message.strip()
        print(f"\n[RAG] Question: {q} | Allow DB: {allow_db}")

        # ── Step 1: Fast keyword routing ──────────────────────────────────────
        route = _route_question(q)
        print(f"  [Router] Keyword decision: {route}")

        if not allow_db:
            if route == "DB":
                return "🔒 **Access Denied:** You need SAP B1 database access permission from an admin to query business data. Please request access from the dashboard.", [], True
            answer, sources = self._gemini_answer(q, gemini_module)
            return answer, sources, False

        if route == "GENERAL":
            answer, sources = self._gemini_answer(q, gemini_module)
            return answer, sources, False

        # ── Step 2: If DB or UNSURE, ask Groq to generate SQL ─────────────────
        if not self._gc:
            print("  [RAG] No Groq client — falling back to Gemini")
            answer, sources = self._gemini_answer(q, gemini_module)
            return answer, sources, False

        system = SQL_SYSTEM.format(max_rows=self.cfg.MAX_ROWS)
        user_p = SQL_USER.format(schema=SAP_SCHEMA, question=q)

        raw_sql = _call_groq(self._gc, self._gm, system, user_p, max_tokens=600)
        print(f"  [Groq→SQL] {raw_sql[:120]}")

        # Groq says it's general chat
        if not raw_sql or "GENERAL_CHAT" in raw_sql.upper():
            print("  [Router] Groq says GENERAL_CHAT → routing to Gemini")
            answer, sources = self._gemini_answer(q, gemini_module)
            return answer, sources, False

        # Not a SELECT — treat as general
        if not re.match(r"^\s*SELECT\b", raw_sql.strip(), re.IGNORECASE):
            print("  [Router] Not a SELECT → routing to Gemini")
            answer, sources = self._gemini_answer(q, gemini_module)
            return answer, sources, False

        # ── Step 3: Execute SQL ───────────────────────────────────────────────
        if not self._sql:
            return "⚠️ Database is not connected. Please check SQL Server.", [], True

        rows, err = self._sql.run(raw_sql)

        # ── Step 4: Auto-fix on error (up to 2 fix attempts) ─────────────────
        for fix_attempt in range(2):
            if not err:
                break
            print(f"  [RAG] SQL error — fix attempt {fix_attempt + 1}: {err[:80]}")
            fix_sys  = FIX_SYSTEM
            fix_user = FIX_USER.format(
                schema=SAP_SCHEMA[:1200],
                question=q,
                sql=raw_sql,
                error=err,
            )
            fixed_sql = _call_groq(self._gc, self._gm, fix_sys, fix_user, max_tokens=600)
            if fixed_sql and re.match(r"^\s*SELECT\b", fixed_sql.strip(), re.IGNORECASE):
                rows, err = self._sql.run(fixed_sql)
                raw_sql = fixed_sql
            else:
                break

        if err:
            return (
                f"⚠️ Sorry, I couldn't retrieve data for: **{q}**\n"
                f"Database error: `{err}`\n"
                f"Please rephrase your question or check if the data exists.",
                [], True
            )

        if not rows:
            return f"No records found for: **{q}**", [], True

        # ── Step 5: Format response ───────────────────────────────────────────
        sources = [{
            "title": f"SAP B1 › {q[:60]}",
            "url":   f"db://{self.cfg.SERVER}/{self.cfg.DATABASE}",
            "rows":  len(rows),
        }]

        # Small result → Groq gives a natural-language summary + table
        if len(rows) <= 5:
            data_str = str([
                {k: _fmt_val(k, v) for k, v in r.items()}
                for r in rows
            ])
            summary = _call_groq(
                self._gc, self._gm,
                ANSWER_SYSTEM,
                ANSWER_USER.format(question=q, data=data_str),
                max_tokens=200,
                temperature=0.3,
            )
            table_html = _format_table(rows)
            return (f"{summary}\n\n{table_html}" if summary else table_html), sources, True

        # Large result → HTML table only (zero extra Groq tokens)
        return _format_table(rows), sources, True

    # ── Routing helpers ───────────────────────────────────────────────────────

    def _gemini_answer(self, q: str, gemini_module) -> tuple:
        """Route to Gemini for general questions."""
        if gemini_module and hasattr(gemini_module, "chat"):
            print(f"  [Router] → Gemini")
            return gemini_module.chat(q)

        # Fallback: use Groq for general answer if Gemini not available
        if self._gc:
            print("  [Router] Gemini not available — using Groq for general answer")
            ans = _call_groq(
                self._gc, self._gm,
                "You are a helpful, friendly assistant. Answer any question clearly and accurately.",
                q,
                max_tokens=500,
                temperature=0.7,
            )
            return ans or "I couldn't find an answer.", []

        return (
            "⚠️ No AI module available to answer general questions. "
            "Please configure Gemini module.",
            [],
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def databases(self) -> list:
        return [self.cfg.DATABASE]

    def table_count(self) -> int:
        return 22

    def refresh_schema(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
    rag = RAGModule(api_key=API_KEY)
    tests = [
        "total revenue this year",
        "top 5 customers by sales",
        "which items are below minimum stock?",
        "show open invoices above 50000",
        "what is the weather in pune",   # → should say GENERAL_CHAT
        "tell me a joke",                # → should say GENERAL_CHAT
        "what is 25 * 48",              # → should say GENERAL_CHAT
    ]
    for q in tests:
        print(f"\n{'='*60}\nQ: {q}")
        ans, src = rag.ask(q)
        print(ans[:300])