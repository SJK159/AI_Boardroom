# AI for the Boardroom

A multi-agent AI system that analyzes company data and delivers decision support through a
boardroom-style UX. Specialist AI agents act as board personas (Finance, Sales, Sentiment,
Operations, Growth, Risk, Compliance/HR), each equipped with domain-specific tools. A boss
agent dynamically decides which specialists are relevant to a given query, invokes them in
parallel, and synthesizes their findings into a final recommendation — **explicitly preserving
disagreement** between agents rather than flattening it into false consensus.

This is a portfolio/agency-demo project built incrementally, one component at a time.

---

## 1. The Concept (High Level)

Picture a real corporate boardroom. A question comes in — "should we expand into a new
region?", "why did Q3 margins slip?" — and different department heads each answer from their
own lens: Finance talks numbers, Sales talks pipeline, Operations talks fulfillment risk. They
don't always agree, and a good chairperson doesn't force fake consensus — they surface the
disagreement and make a call.

This project builds that as software:

- **Specialist agents** are the department heads. Each one only knows its own domain (Finance
  doesn't reason about delivery logistics) and only has tools relevant to that domain.
- **The boss agent** is the chairperson. It reads the incoming question, decides which
  specialists are actually relevant (not all 7 are needed for every query), asks them to
  investigate in parallel, and then writes a memo that synthesizes their findings —
  **including where they disagree**.
- **Every claim is sourced.** No specialist is allowed to state a number without saying which
  tool/query produced it. This isn't a nice-to-have — it's the difference between a "trust me"
  chatbot and something a real board could use to make a decision.
- **Nothing acts on its own.** The system produces a *recommendation*. A human has to
  accept/reject/modify it before it becomes a decision. This is a governance decision, not a
  technical limitation.

### Why this design (not just "one big LLM prompt")

A single LLM answering "how's the business doing?" in one shot has no way to show its work,
no way to disagree with itself, and no way to keep a growing tool-belt organized as the system
scales to 7 domains × 7 tools each (49 tools). Splitting into specialist agents means:

1. Each agent's tools stay small and testable in isolation.
2. The boss agent's job (synthesis + disagreement detection) stays separate from data-fetching,
   so it can use a stronger/more expensive model since it only runs once per session instead of
   once per tool call.
3. Adding an 8th specialist later doesn't touch the other 7.

---

## 2. Architecture (Low Level)

Five layers, data flows bottom-up into agents, then top-down as a synthesized answer:

```
┌─────────────────────────────────────────────────────────────┐
│  5. UX Layer           React + Socket.io (not yet built)     │
├─────────────────────────────────────────────────────────────┤
│  4. Agents             LangGraph supervisor pattern           │
│                        7 specialists + 1 boss agent           │
│                        (Finance Agent built; others pending)  │
├─────────────────────────────────────────────────────────────┤
│  3. SLM (briefing)     Ollama + qwen2.5:7b-instruct            │
│                        (not yet built)                        │
├─────────────────────────────────────────────────────────────┤
│  2. NLP/DL             Entity extraction, sentiment, anomaly  │
│                        (not yet built — Finance Agent does    │
│                        its own lightweight anomaly detection  │
│                        directly for now)                      │
├─────────────────────────────────────────────────────────────┤
│  1. RAG / Data         Databricks: Delta Lake + Unity Catalog │
│                        9 tables live, Vector Search pending   │
└─────────────────────────────────────────────────────────────┘
```

**Current build status**: layer 1 (data) is live, layer 4 has its first specialist (Finance)
working end-to-end against real data. Layers 2, 3, 5 and the boss agent are not built yet.

### Query-time flow (target — boss agent not built yet)

```
User query
  → Boss Agent parses intent
  → Boss Agent selects relevant specialist agents
  → Selected agents run in PARALLEL, each invoking its own tools
  → Each agent produces a validated AgentBriefing
  → Boss Agent collects all briefings, detects disagreement,
    synthesizes into a board memo with citations
  → Streamed to UX layer
```

### What exists right now (this session's flow)

```
scripts/run_finance_agent.py
  → FinanceAgent.run(query)
      → analyze() calls all 7 finance tools via _call_tool()
          → each tool runs SQL against Delta tables through DatabricksClient
          → every call is logged as a ToolCallRecord (success/failure, timing)
      → results become Finding objects (claim + source + confidence)
  → returns a validated AgentBriefing (Pydantic-checked)
```

---

## 3. Data

**Source**: [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(public, Kaggle). ~100k real orders (2016–2018), pulled directly via the Kaggle API.

**Where it lives**: uploaded to a Databricks Unity Catalog Volume, then materialized as managed
Delta tables — `workspace.olist.*`:

| Table | Rows (approx) | Used by |
|---|---|---|
| `orders` | 99,441 | Finance, Sales, Operations, Risk |
| `order_items` | ~112k | Finance, Sales, Growth |
| `order_payments` | ~104k | Finance, Risk |
| `order_reviews` | ~99k | Sentiment (RAG, pending) |
| `customers` | 99,441 | Sales, Growth, Risk |
| `sellers` | ~3,100 | Operations, Risk, Compliance |
| `products` | ~32,900 | Sales, Growth |
| `geolocation` | ~1M | Growth, Operations |
| `product_category_translation` | 71 | Sales, Growth (PT→EN mapping) |

**Known data limitation** (important — affects Finance Agent design below): the dataset has
**no cost/COGS field** anywhere. `products` only has physical attributes (weight, dimensions,
photo count), not unit cost. There is also **no explicit payment-failure or refund flag** —
`order_status` values (`canceled`, `unavailable`) are the closest proxies. Every Finance tool
that touches these gaps says so explicitly in its output rather than fabricating a number —
this is a deliberate governance choice (see section 6 of the project spec, `CLAUDE.md`).

Supplementary institutional documents (fictional HR policy, vendor contracts, company
registration for a wrapper entity "Olist Inc.") are planned for the Compliance/HR agent but
**not built yet** — they'll be created when that agent is built, not fabricated in advance.

---

## 4. What's Built So Far

### 4.1 Databricks setup

- Workspace connected via `DATABRICKS_HOST` + `DATABRICKS_TOKEN` (personal access token)
- Unity Catalog: catalog `workspace`, schema `olist`
- Data uploaded to a managed **Volume** (`workspace.olist.raw_files`) — the workspace has
  public DBFS root disabled, so Volumes are the correct/modern path anyway
- All 9 Delta tables created via a **serverless SQL warehouse** using `read_files()` +
  `CREATE TABLE ... AS SELECT` (no cluster needed — the workspace has no cluster compute
  policy configured, so this was also the only viable path)
- Script: [`scripts/ingest_to_databricks.py`](scripts/ingest_to_databricks.py)

### 4.2 Backend scaffolding

```
backend/
├── schemas/          Pydantic models — the validated contract every agent output must pass
│   ├── enums.py          AgentType (Finance, Sales, Sentiment, Operations, Growth, Risk, Compliance)
│   ├── findings.py        Finding, ToolCallRecord
│   ├── briefing.py        AgentBriefing (what a specialist agent returns)
│   ├── recommendation.py  Dissent, BoardRecommendation (what the boss agent will return)
│   └── governance.py      GovernanceLog (full session audit trail)
├── config/
│   └── settings.py    Centralized env loading (pydantic-settings, reads .env once)
├── db/
│   └── databricks_client.py   Thin SQL wrapper — the ONLY file that talks to the Databricks SDK
├── agents/
│   ├── base_agent.py   SpecialistAgent abstract base class (see below)
│   └── finance/         First specialist agent, fully implemented
│       ├── tools.py        7 tools, pure functions, each returns raw computed data
│       └── agent.py        FinanceAgent — turns tool output into Finding objects
```

**Why schemas are split by file, not one giant `schemas.py`**: `Finding`/`ToolCallRecord` →
`AgentBriefing` → `BoardRecommendation`/`Dissent` → `GovernanceLog` form a dependency chain.
Splitting them makes that chain visible and keeps each file reviewable on its own.

**Why `DatabricksClient` is a single chokepoint**: no agent or tool file imports the Databricks
SDK directly. They all call `db.query(sql)`. This means swapping the SQL warehouse for
`databricks-connect` or a cluster later touches exactly one file, not 49 tool functions.

### 4.3 `SpecialistAgent` base class

Every specialist agent (Finance built, 6 more to come) inherits from this. It exists to
guarantee governance is structural, not something each agent remembers to do:

```python
class SpecialistAgent(ABC):
    agent_type: AgentType

    def _call_tool(self, tool_name, func, **kwargs):
        # times the call, catches exceptions, records a ToolCallRecord
        # regardless of what the tool does internally

    @abstractmethod
    def analyze(self, query: str) -> list[Finding]:
        # subclass implements domain logic here

    def run(self, query: str) -> AgentBriefing:
        # calls analyze(), wraps everything into a validated AgentBriefing
```

Every tool call — success or failure, timing included — is captured automatically. A
specialist agent physically cannot skip logging a tool call, because `_call_tool()` is the only
path to invoking one. This is the "logging middleware wrapping every agent call" requirement
from the project governance spec, implemented at the base-class level instead of per-agent.

### 4.4 Finance Agent — all 7 tools, live against real data

| Tool | What it computes | Data reality check |
|---|---|---|
| `calculate_margin_trend()` | Monthly contribution margin (`price - freight_value`), trend direction | True gross margin needs COGS (unavailable) — uses freight as the only real per-order cost signal, flagged as a proxy |
| `detect_revenue_anomalies()` | Daily revenue, flags days beyond N std devs from the mean (z-score) | Computed directly, no proxy needed |
| `payment_failure_rate()` | % of orders `canceled`/`unavailable` | No explicit payment-failure field in source — flagged as a proxy |
| `calculate_cogs()` | — | **Explicitly returns "not available"** rather than fabricating a number. `products` table has no cost field. This is the CLAUDE.md-mandated "flag rather than force a fake signal" behavior in practice |
| `cash_flow_forecast()` | Linear-trend projection of net cash-in for N future months | Simple linear regression (`numpy.polyfit`) over monthly totals — not a full AR/AP cash-flow model, flagged as such |
| `refund_impact_analysis()` | Payment value at risk from canceled/unavailable orders | No explicit refund field — flagged as a proxy |
| `revenue_concentration()` | Seller revenue concentration: top-10 share + Herfindahl-Hirschman Index | Computed directly from `order_items` |

**Verified output** (from a live run against `workspace.olist.*`):

```
Agent: Finance
Tool calls: 7 (7 succeeded)

[INFO]     (0.75) Contribution margin is improving across the last 12 months
[WARNING]  (0.85) 13 daily revenue anomalies detected out of 614 days analyzed
[INFO]     (0.60) Order failure rate (canceled/unavailable proxy) is 1.24% of 99441 orders
[WARNING]  (1.00) COGS cannot be calculated - source data has no unit cost field
[INFO]     (0.55) Next month's projected cash-in is 1177859.93
[INFO]     (0.60) 1.68% of total payment value (269735.11) is at risk from canceled/unavailable orders
[INFO]     (0.90) Revenue concentration is low - top 10 sellers hold 13.2% of revenue (HHI 36.0)
```

Every finding carries a `confidence` score (lower for proxy-based tools like
`payment_failure_rate` at 0.6, higher for directly-computed ones like `revenue_concentration`
at 0.9) and a `source` field naming the exact tool that produced it — this is the traceability
requirement from the governance spec, not an afterthought.

Run it yourself: `python scripts/run_finance_agent.py`

---

## 5. Tools & Stack

| Purpose | Tool/Library | Notes |
|---|---|---|
| Data warehouse | **Databricks** (Delta Lake, Unity Catalog) | Managed Volumes used instead of DBFS root (disabled on this workspace by policy) |
| SQL execution | **Databricks serverless SQL warehouse** | No cluster compute available on this workspace — warehouse is also cheaper/faster to start |
| Data source | **Kaggle API** | Pulled Olist dataset directly, no manual download |
| Schema validation | **Pydantic v2** | Every agent output validated before being treated as trustworthy |
| Settings | **pydantic-settings** | Single source of truth for env vars, loaded from `.env` |
| Numerics | **NumPy** | z-score anomaly detection, linear regression for cash-flow forecast |
| Language | **Python 3.14** | |
| Agent orchestration (planned) | **LangGraph** | Supervisor/multi-agent pattern — boss agent not built yet |
| SLM (planned) | **Ollama + qwen2.5:7b-instruct** | Briefing compression layer — not built yet |
| Frontend (planned) | **React (MERN) + Socket.io** | Not built yet |

---

## 6. Governance (Built-In, Not Bolted On)

Four pillars from the project spec, and what's actually implemented today:

| Pillar | Status |
|---|---|
| **Transparency** | Every `Finding` carries `confidence` + `severity`. Proxy-based tools (payment failure, refunds, margin) explicitly say so in their output rather than presenting estimates as facts. |
| **Traceability** | Every `Finding.source` names the exact tool that produced it. Every tool call — args, output, timing, success/failure — is captured in `ToolCallRecord`, automatically, via the base class. |
| **Human Oversight** | `BoardRecommendation.requires_human_approval` defaults to `True` (schema-level, not agent-level — can't be silently skipped). Boss agent not built yet, so this isn't exercised end-to-end yet, but the schema enforces it going forward. |
| **Accountability** | `GovernanceLog` schema exists to capture full session audit trails (query, agents invoked, findings, model versions, human decision). Not yet wired into a persistence layer (MongoDB — planned, not built). |

---

## 7. Environment Setup

Requires a `.env` file (gitignored, not committed) with:

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_KEY=your_kaggle_key
```

```bash
pip install -r requirements.txt
python scripts/ingest_to_databricks.py    # one-time: pull data + create Delta tables
python scripts/run_finance_agent.py       # run the Finance Agent against live data
```

---

## 8. What's Next

Per the build order in `CLAUDE.md`:

1. ~~Repo scaffolding + Pydantic schemas + base `SpecialistAgent` class~~ ✅
2. ~~Databricks connection + Olist data ingestion into Delta tables~~ ✅
3. ~~Finance Agent end-to-end, all 7 tools~~ ✅
4. **Boss agent skeleton (LangGraph supervisor), wired to Finance Agent** ← next
5. Remaining 6 specialist agents (Sales, Sentiment, Operations, Growth, Risk, Compliance/HR)
6. RAG layer (Vector Search) for Sentiment Agent + simulated institutional docs for Compliance
7. Governance logging middleware persistence + MongoDB
8. MERN frontend + streaming + WebSocket agent status
9. Eval suite
10. Deployment containerization

Full project spec, including all 7 agents' complete tool lists and the schema reference, lives
in [`CLAUDE.md`](CLAUDE.md).
