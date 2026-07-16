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

**Current build status**: layer 1 (data) is live. Layer 4 is complete — all 7 specialists
working end-to-end against real data, orchestrated by a working boss agent. Layers 2, 3, and 5
are not built yet.

### Query-time flow (as built — all 7 specialists wired up)

```
User query
  → Boss Agent (LangGraph graph, Groq-hosted LLM) parses intent
  → Boss Agent selects relevant specialist agents from its registry
      (Finance, Sentiment, Sales, Operations, Growth, Risk, Compliance/HR —
       the registry is the single place specialist availability is declared)
  → Selected agents run in PARALLEL (thread pool), each invoking its own tools
  → Each agent produces a validated AgentBriefing
  → Boss Agent collects all briefings, detects disagreement,
    synthesizes into a board memo with citations
  → Returns a validated BoardRecommendation (requires_human_approval = True)
```

### What exists right now

```
notebooks/09_compliance_agent_demo.ipynb  (or any of notebooks/03-09)
  → BossAgent.run(query)
      → select_specialists node: LLM decides which registered specialists apply
          → (if none apply, short-circuits to a "no relevant agent" response)
      → run_specialists node: parallel .run(query) calls, one per selected specialist
          → e.g. FinanceAgent.run(query)
              → analyze() calls all of that agent's tools via _call_tool()
                  → each tool runs SQL (or, for Compliance, local document lookups)
                    through DatabricksClient
                  → every call is logged as a ToolCallRecord (success/failure, timing)
              → results become Finding objects (claim + source + confidence)
              → returns a validated AgentBriefing
      → synthesize node: LLM writes the board memo from all briefings,
        flags dissent between agents, sets overall confidence + action items
  → returns a validated BoardRecommendation
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
| `order_reviews` | 99,249 | Sentiment (full search RAG pending — see 4.6) |
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

Supplementary institutional documents (fictional HR policy, vendor contract, company
registration for a wrapper entity "Olist Inc.") are **built** — created alongside the
Compliance/HR agent per CLAUDE.md's instruction to build them "once we're hands-on with the
data," not fabricated in advance. They live at `backend/agents/compliance/documents/` as local
markdown, parsed by section header — see section 4.11.

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
- Notebook: [`notebooks/01_data_ingestion.ipynb`](notebooks/01_data_ingestion.ipynb)

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
│   ├── finance/         First specialist agent, fully implemented
│   │   ├── tools.py        7 tools, pure functions, each returns raw computed data
│   │   └── agent.py        FinanceAgent — turns tool output into Finding objects
│   ├── sentiment/        Second specialist agent, fully implemented
│   │   ├── tools.py        7 tools — review sentiment, complaints, regional/photo correlation
│   │   └── agent.py        SentimentAgent — turns tool output into Finding objects
│   ├── sales/            Third specialist agent, fully implemented
│   │   ├── tools.py        8 tools — revenue trend, AOV, category/seller ranking, cross-sell
│   │   └── agent.py        SalesAgent — turns tool output into Finding objects
│   ├── operations/       Fourth specialist agent, fully implemented
│   │   ├── tools.py        7 tools — delivery delay, seller reliability, fulfillment bottlenecks
│   │   └── agent.py        OperationsAgent — turns tool output into Finding objects
│   ├── growth/           Fifth specialist agent, fully implemented
│   │   ├── tools.py        7 tools — regional/category growth, acquisition, product-gap proxy
│   │   └── agent.py        GrowthAgent — turns tool output into Finding objects
│   ├── risk/             Sixth specialist agent, fully implemented
│   │   ├── tools.py        7 tools — cancellations, disputes, churn, concentration, fraud signals
│   │   └── agent.py        RiskAgent — turns tool output into Finding objects
│   ├── compliance/       Seventh specialist agent, fully implemented - ROSTER COMPLETE
│   │   ├── documents/      3 fictional institutional docs (registration, HR policy, contract)
│   │   ├── document_loader.py  Section-based markdown parser (clause-addressable text)
│   │   ├── tools.py        7 tools — doc fact retrieval + the flagship SLA cross-reference
│   │   └── agent.py        ComplianceAgent — turns tool output into Finding objects
│   └── boss/            Orchestrator — no domain tools, LLM-driven
│       ├── registry.py     AVAILABLE_SPECIALISTS — the one place that grows per new agent
│       ├── llm_outputs.py  Structured-output shapes the boss LLM must return
│       ├── prompts.py      Selection + synthesis prompt templates
│       ├── state.py        BossState — shared state across the LangGraph graph
│       └── graph.py        BossAgent — builds and runs the 3-node LangGraph graph
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

Run it yourself: [`notebooks/02_finance_agent_demo.ipynb`](notebooks/02_finance_agent_demo.ipynb)

### 4.5 Boss Agent — LangGraph orchestration, wired to Finance

The boss agent has no domain tools of its own. It's a 3-node LangGraph graph:

```
select_specialists  →  run_specialists  →  synthesize
       │
       └─ (if none selected) → no_relevant_agents → END
```

- **`select_specialists`**: an LLM call reads the query plus the roster from
  `registry.AVAILABLE_SPECIALISTS` (currently: Finance only) and returns which specialists are
  actually relevant, via structured output (`AgentSelection` Pydantic model — not a raw string
  the code has to parse). If none apply, the graph short-circuits rather than forcing an
  irrelevant specialist to answer.
- **`run_specialists`**: invokes `.run(query)` on every selected specialist through a
  `ThreadPoolExecutor` — parallel by construction, even though there's only one specialist to
  parallelize today. Adding specialist #2 doesn't touch this node.
- **`synthesize`**: a second LLM call takes every `AgentBriefing`, writes the board memo, and
  is explicitly prompted to flag disagreement between agents as a `Dissent` rather than
  smoothing it into consensus. Also structured output (`SynthesisOutput`).

**Why a registry, not a fixed agent list**: `AVAILABLE_SPECIALISTS` in
`backend/agents/boss/registry.py` is the *only* place the boss agent knows what specialists
exist. The selection LLM is only ever shown agents actually in that dict, so it can never route
a query to a specialist that isn't implemented yet. Adding Sales next session is one dict entry
here — nothing else in the boss agent changes.

**Why Groq, not a paid hosted API**: CLAUDE.md's spec calls for "a stronger hosted LLM... NOT
the local SLM" for the boss agent, since it only runs once per session rather than once per
tool call. That's still true with Groq — the point is a *hosted*, *capable* model separate from
the lightweight local SLM layer, not necessarily the most expensive one. Groq's free tier
serves GPT OSS 20B fast enough for orchestration and synthesis, without burning money on
every test run during active development. This can be swapped for Anthropic/OpenAI later by
changing one line in `graph.py` — nothing else depends on which provider `ChatGroq` becomes.

Run it yourself: [`notebooks/03_boss_agent_demo.ipynb`](notebooks/03_boss_agent_demo.ipynb) —
includes a second query deliberately outside Finance's domain, to demonstrate that agent
selection is a real decision and not a rubber stamp.

### 4.6 Sentiment Agent — second specialist, first real dissent

| Tool | Computes | Data reality |
|---|---|---|
| `search_reviews` | Keyword match over review text | Proxy for real semantic search — Vector Search (build order step 6) isn't built yet, and this only works well with Portuguese-language queries |
| `sentiment_score_by_product` | Avg `review_score` (1-5) per product, top/bottom N | `review_score` is already a direct sentiment signal — no NLP needed |
| `flag_negative_trend` | Monthly share of score <=2 reviews, trend direction | Computed directly |
| `extract_common_complaints` | Word-frequency on negative review text | Raw Portuguese, basic stopword filtering only — real topic extraction needs the NLP/DL layer (step 2, not built) |
| `review_response_time_correlation` | Correlation between survey-answer latency and score | **Not seller response time** — `review_answer_timestamp` is when the *customer* answered Olist's satisfaction survey. No seller-response field exists in this dataset; the tool is named and flagged accordingly rather than implying something the data can't support |
| `sentiment_by_region` | Avg score by customer state | Computed directly |
| `photo_review_analysis` | Correlation between product *listing* photo count and score | Olist reviews have no photo attachments — uses `products.product_photos_qty` as the closest proxy |

**A real data-quality bug found and fixed while building this agent**: ~0.07% of
`order_reviews` rows (73 of 99,249) have review text containing both embedded newlines and
doubled-quote escaping (`""word""`) together — an edge case Databricks' `multiLine` CSV reader
doesn't fully resolve, which shifted text into the `review_creation_date` and even
`review_score` columns for those rows. Two things were done about it:

1. **Fixed at the source**: `notebooks/01_data_ingestion.ipynb` now reads `order_reviews` with
   `multiLine => true`, which alone dropped a separate, larger issue — the *original* ingestion
   (without `multiLine`) had inflated `order_reviews` to 104,162 rows by splitting reviews with
   embedded newlines into multiple physical rows. The corrected count, 99,249, is much closer
   to the dataset's documented ~99,224.
2. **Defended in the tools**: every query touching `review_score` or the review dates uses
   `try_cast(...)`, which returns `NULL` for the ~73 still-malformed rows instead of crashing —
   `NULL` is automatically skipped by SQL aggregates (`AVG`, `SUM`) and `WHERE` filters, so no
   separate `IS NOT NULL` bookkeeping was needed once `try_cast` was in place.

**First demonstrated dissent** — with Finance and Sentiment both registered, asking *"Are
customers happy, and is that reflected in the numbers?"* correctly invoked both agents in
parallel, and the synthesis flagged a genuine conflict rather than picking a side:

> Finance reports an improving contribution margin and low revenue concentration, yet
> Sentiment indicates flat negative review share and a product with a 1.0/5 rating. The two
> perspectives conflict on the implication that customer happiness is reflected in financial
> performance.

This is the mechanism CLAUDE.md's core design goal depends on — disagreement surfaced as a
structured `Dissent`, not smoothed into false consensus.

Run it yourself: [`notebooks/04_sentiment_agent_demo.ipynb`](notebooks/04_sentiment_agent_demo.ipynb)

### 4.7 Sales Agent — third specialist

| Tool | Computes | Data reality |
|---|---|---|
| `query_revenue_by_period` | Revenue + order count per day/week/month, trend direction | **Excludes trailing partial periods from the trend calc** — see the bug below |
| `calculate_aov` | Overall + monthly average order value | Computed directly |
| `sales_by_category` | Revenue and items sold per product category | Computed directly |
| `seller_sales_ranking` | Top sellers by revenue/order volume | Volume lens only — deliberately overlaps with Operations' future `seller_performance_score` (reliability lens); CLAUDE.md calls this overlap intentional, not a bug |
| `conversion_funnel_analysis` | Order status breakdown, drop-off rate | **Not a true top-of-funnel conversion** — Olist has no browse/cart data, so this proxies the *fulfillment* funnel (placed → delivered vs. canceled/unavailable), flagged as such |
| `repeat_purchase_rate` | Share of customers with 2+ orders | Uses `customer_unique_id` (person-level), not `customer_id` — Olist issues a new `customer_id` per order, so using it here would silently read as ~0% repeat rate |
| `seasonal_sales_pattern` | Revenue by calendar month, aggregated across years | Only 2016-2018 in the data — not enough years to cleanly separate seasonality from year-over-year growth |
| `cross_sell_opportunities` | Category pairs frequently bought together | Simple co-occurrence within an order, not a trained recommendation model |

**A second real data bug, caught the same way as the Sentiment one — by checking a suspicious
number instead of trusting it.** The naive "last 12 months" window initially reported revenue
as **declining**. The raw monthly order counts explain why:

| Month | Orders |
|---|---|
| 2018-08 | 6,512 |
| 2018-09 | 16 |
| 2018-10 | 4 |

Olist's data collection stops mid-way through September 2018 — the tail of the window is a
collection cutoff, not a demand collapse. `query_revenue_by_period()` now detects trailing
periods whose order count falls far below the window's median and excludes them from the trend
calculation (while still returning them in the raw data via `excluded_partial_periods`), rather
than reporting a false decline. This is exactly the class of error a single LLM summarizing raw
numbers in one shot would likely miss silently — a concrete case for why tool-backed, code-
verified findings matter over pure LLM reasoning over raw data, which is the whole premise this
project is built on.

**Known synthesis limitation, observed live**: with 3 specialists now registered, one board
memo briefly misattributed `query_revenue_by_period` (a Sales-only tool) to Finance in its
citation. Not a data pipeline bug — a citation slip from GPT OSS 20B during synthesis. Worth
tracking as the roster grows; the Llama 3.3 70B / GPT OSS 120B upgrade path discussed earlier
is the lever if citation accuracy needs to improve.

Run it yourself: [`notebooks/05_sales_agent_demo.ipynb`](notebooks/05_sales_agent_demo.ipynb)

### 4.8 Operations/Logistics Agent — fourth specialist, second demonstrated dissent

| Tool | Computes | Data reality |
|---|---|---|
| `calculate_delivery_delay` | Avg delay + late-delivery rate vs. Olist's own delivery estimate | Computed directly |
| `seller_performance_score` | Seller reliability: on-time %, avg delay | Reliability lens only — deliberately overlaps with Sales' `seller_sales_ranking` (revenue/volume), a different lens on the same sellers, per CLAUDE.md |
| `flag_late_shipments` | Mild vs. severe lateness buckets | Computed directly |
| `shipping_cost_analysis` | Freight cost, interstate vs. intrastate | Logistics-efficiency framing — deliberately distinct from Finance's margin-impact framing of the same `freight_value` field |
| `carrier_performance_comparison` | — | **Explicitly returns "not available"** — Olist's schema has no carrier/shipping-company identifier anywhere, so there's nothing to compare across carriers. Third tool across the roster (after Finance's `calculate_cogs`) to hit a hard data wall and say so rather than approximate something misleading |
| `fulfillment_bottleneck_detection` | Avg duration of each fulfillment stage | Uses fractional-day precision, not whole-day `datediff`, so same-day stages like approval still show meaningful averages instead of rounding to zero |
| `estimated_vs_actual_delivery_accuracy` | MAE + bias of Olist's own delivery estimate | Bias sign is a real signal, not just an error metric: -11.88 days observed live means Olist's estimate is conservative and deliveries tend to arrive early (under-promise, over-deliver) |

**Second demonstrated dissent** — with Operations and Sentiment both registered, asking
*"Are delivery problems hurting customer satisfaction?"* correctly invoked both, and again the
synthesis surfaced a real conflict instead of picking a side:

> Operations reports 8.11% late deliveries and 9.33-day transit times, yet Sentiment finds a
> flat negative review share over the last 12 months and no explicit mentions of delivery
> problems in reviews. The two findings conflict regarding the impact of delivery performance
> on customer satisfaction.

Two dissents now demonstrated across two independent specialist pairings (Finance/Sentiment,
Operations/Sentiment) — the mechanism isn't a one-off, it holds as the roster grows.

Run it yourself: [`notebooks/06_operations_agent_demo.ipynb`](notebooks/06_operations_agent_demo.ipynb)

### 4.9 Growth Agent — fifth specialist, and a real Groq reliability fix

| Tool | Computes | Data reality |
|---|---|---|
| `regional_sales_breakdown` | Revenue + order count by customer state | Computed directly |
| `market_expansion_signals` | States with strongest revenue growth (recent 6mo vs. prior 6mo) | Filters out small-base noise — see below |
| `category_growth_rate` | Category revenue growth, same 6mo/6mo window | Same small-base filter |
| `customer_acquisition_trend` | New customers per month (first-order date, person-level) | Trailing-cutoff aware (reuses the Sales-agent fix) |
| `underperforming_region_diagnosis` | States with declining revenue | Same small-base filter, inverse direction |
| `new_seller_onboarding_rate` | New sellers per month (first-order date) | Trailing-cutoff aware |
| `product_gap_analysis` | Revenue-per-seller by category (supply/demand proxy) | **Explicitly not true product-gap analysis** — see below |

**Shared infrastructure**: every trend tool here calls `_get_analysis_cutoff()` once, which
reuses the same median-based heuristic Sales' `query_revenue_by_period` introduced to detect
Olist's September 2018 data-collection cutoff, rather than re-deriving the fix per tool.

**A third real bug — small-base noise, not a data-quality defect this time.** The first version
of `market_expansion_signals()` reported **RR at 827.96% growth** as the strongest signal. The
raw numbers: RR's prior-period revenue was R$656.61 — a percentage computed off a near-empty
base isn't a real trend, it's noise. `category_growth_rate()` had the identical problem: music
showed 4849.52% growth off a R$79.87 base. Both tools (plus `underperforming_region_diagnosis`,
same issue in the opposite direction) now filter out entries below a minimum prior-period
revenue threshold before ranking — the same `min_reviews`/`min_orders` guard pattern already
used in Sentiment and Operations for this exact class of problem.

**`product_gap_analysis`** is the tool CLAUDE.md explicitly calls out in advance as the
weakest fit for this dataset: true product-gap/whitespace analysis needs external market data
— visibility into demand for products never listed on the platform at all, which a
transactions-only dataset cannot provide. What's implemented is a supply/demand imbalance proxy
*within* existing categories (revenue-per-seller, flagging under- vs. over-supplied
categories), and the tool's own output says so rather than presenting it as real whitespace
analysis.

**A real boss-agent reliability bug, found and fixed**: while running this notebook, the boss
agent's synthesis step failed with `BadRequestError: Tool call validation failed: attempted to
call tool 'synthesisOutput' which was not in request.tools`. This is a known reliability quirk
of smaller open-weight models (GPT OSS 20B here) under strict structured-output/function-calling
— the model occasionally hallucinates a slightly mismatched tool name. It reproduced
intermittently (failed once, succeeded on an immediate identical retry), which is the signature
of a transient model-level issue rather than a schema or prompt bug. `BossAgent` now wraps both
structured-output LLM calls (`_select_specialists`, `_synthesize`) in a retry helper
(`_invoke_with_retry`, up to 3 attempts with backoff) rather than letting one flaky call break
the whole pipeline — standard practice for a known-flaky external dependency, not a workaround
for something actually wrong in our code.

**Cross-agent synthesis, no forced dissent this time** — asking *"Which regions should we
invest in for growth, and can we support that operationally?"* correctly invoked Growth and
Operations together, and the synthesis didn't fabricate a conflict where there wasn't one — it
combined Growth's expansion signal (PI, +68.7%) with Operations' capacity constraints (9.33-day
transit bottleneck, higher interstate freight cost) into a genuinely useful "invest here, but
fix this operational gap first" recommendation. Complementary, not contradictory — a good sign
the synthesis isn't manufacturing dissent just because it's primed to look for it.

Run it yourself: [`notebooks/07_growth_agent_demo.ipynb`](notebooks/07_growth_agent_demo.ipynb)

### 4.10 Risk Agent — sixth specialist, a real statistics bug caught before it shipped

| Tool | Computes | Data reality |
|---|---|---|
| `cancellation_trend` | Monthly cancellation rate + trend | Trailing-cutoff aware (reuses the Sales-agent fix) |
| `flag_payment_disputes` | Canceled orders with payment already collected | Proxy — Olist has no explicit dispute/chargeback field |
| `seller_churn_risk` | Sellers inactive for N+ months | Computed directly |
| `concentration_risk` | Customer-side revenue concentration (HHI) | Deliberately distinct from Finance's `revenue_concentration` (seller-side) — same HHI methodology, different lens, not a duplicate |
| `fraud_signal_detection` | Top-percentile payment-value outliers | **Switched from z-score to percentile mid-build** — see below |
| `customer_churn_prediction` | Repeat customers overdue for their next order | Heuristic (own historical order gap), not a trained model. "Now" is the dataset's own analysis cutoff, not today's date — computing recency against the real current date would flag every customer as churned, since the data ends in 2018 |
| `regulatory_exposure_check` | — | **Explicitly not available** — Olist's transactional data has zero legal/compliance signals (no consent records, no KYC/AML, no tax jurisdiction data). Real exposure checking depends on the Compliance/HR agent's institutional documents, not built yet |

**A statistics methodology bug, caught by checking the numbers instead of trusting them.** The
first version of `fraud_signal_detection()` used a z-score threshold (`z >= 3.0`) and flagged
**1,724 of 99,440 orders (1.73%)**. Under a normal distribution, `z >= 3` should only catch
about **0.13%** — the flagged rate was over 13x too high. The actual distribution explained why:

| Stat | Value |
|---|---|
| Mean | R$160.99 |
| Median | R$105.29 |
| Skewness | ~9.15 (a normal distribution is 0) |
| Max | R$13,664.08 |

Per-order payment value is heavily right-skewed. Z-score assumes rough symmetry; on skewed
data it systematically over-flags, because "3 standard deviations above the mean" isn't
actually rare when the mean itself is dragged up by a long tail. Fixed by switching to a
**percentile threshold** (top 0.5%), which makes no assumption about distribution shape —
re-running against the same data flagged **498 orders**, almost exactly the expected 0.5% of
99,440. This is the same instinct as the Sales trend bug and the Growth small-base-noise bug:
a number that looks too clean or too extreme gets checked against the raw distribution before
it ships, not after.

**Four-agent synthesis, cleanly organized**: asking *"What should keep us up at night about
this business?"* correctly invoked Risk, Finance, Operations, and Sentiment together. The
resulting memo organized nine distinct risk areas (revenue anomalies, the COGS data gap,
delivery performance, seller churn, payment disputes, regulatory visibility, customer
concentration, product quality, and the carrier-data gap) with correct per-claim tool
citations throughout — the synthesis holds together even as the number of simultaneous
specialists grows.

Run it yourself: [`notebooks/08_risk_agent_demo.ipynb`](notebooks/08_risk_agent_demo.ipynb)

### 4.11 Compliance/HR Agent — seventh and final specialist, roster complete

Architecturally different from the other six: instead of computing metrics from Delta tables,
most of its tools do **precision fact retrieval** against three fictional institutional
documents built specifically for this agent (per CLAUDE.md section 2, a wrapper entity
"Olist Inc." — not real company data): `company_registration.md`, `hr_policy.md`, and
`vendor_contract.md`, stored as local markdown and parsed by section header
(`backend/agents/compliance/document_loader.py`).

| Tool | Computes | Data reality |
|---|---|---|
| `search_policy_docs` | Keyword search across all 3 documents | Proxy for real semantic search — Vector Search with `doc_type` metadata tagging (a separate collection from reviews, per CLAUDE.md) is build-order step 6, not built yet |
| `get_company_registration_info` | All registration fields | Direct structured lookup — exactly the "precision fact-retrieval, not pattern retrieval" CLAUDE.md calls for with these docs |
| `check_contract_clause` | Looks up a clause by topic for a seller | All sellers share ONE standard template — no bespoke per-vendor contracts exist. `vendor` validates the seller_id and is accepted for interface consistency with a future state where addenda might exist |
| `check_policy_compliance` | Looks up whether an HR topic is documented | Direct structured lookup |
| `contract_expiry_tracker` | Sellers approaching their annual contract renewal | Renewal date computed from REAL seller onboarding dates (first order date) + the contract's stated annual cadence |
| `policy_gap_analysis` | Which expected HR topics are missing | **Deliberately finds real gaps** — see below |
| `cross_reference_sla_compliance` | Seller's actual on-time % vs. the contractual SLA threshold | **The flagship cross-agent tool** — see below |

**A deliberately incomplete HR policy.** The document covers 5 of 9 standard topics; four are
missing on purpose — Data Privacy Policy, Anti-Discrimination & Equal Opportunity, Remote Work
Policy, Grievance Procedure. If every topic were present, `policy_gap_analysis()` would always
return "no gaps," which demonstrates nothing. The tool's scope is explicitly bounded: it only
reports documentation coverage, never recommends or makes employment decisions — matching
CLAUDE.md section 6's explicit carve-out for this agent around anything resembling employment
decisions (EU AI Act high-risk category).

**The flagship cross-agent tool, working as designed.** CLAUDE.md calls this tool out
specifically: *"No single agent can answer this alone... the strongest demo of why the
multi-agent design earns its complexity."* `cross_reference_sla_compliance()`:

1. Parses the SLA threshold directly out of the contract text via regex (`no less than 85%`)
   rather than hardcoding a duplicate number, so the check stays in sync if the contract
   document ever changes.
2. Computes a specific seller's actual on-time delivery rate directly against the Delta
   tables, using the same on-time definition as Operations' `calculate_delivery_delay` /
   `seller_performance_score` — kept self-contained per this codebase's convention (no
   cross-package Python import), but the same underlying methodology.
3. Verdicts compliant/breach — something neither agent's tools could answer alone: Operations
   has no notion of a contractual threshold, and Compliance computes no delivery metrics
   anywhere else.

**Verified live**: the platform's highest-volume seller achieved **93.79% on-time delivery**
against an **85.0%** contractual threshold — parsed straight from the document — correctly
verdicted `compliant: True`.

**Roster-complete synthesis** — asking *"Is our top seller compliant with their contract, and
what's the financial exposure if they're not?"* correctly invoked Compliance and Finance
together, verified the seller's SLA compliance, and pulled in Finance's revenue-anomaly and
COGS-gap findings to give a complete (if partially uncertain, due to the still-missing COGS
data) picture of financial exposure — with accurate per-claim tool citations throughout.

Run it yourself: [`notebooks/09_compliance_agent_demo.ipynb`](notebooks/09_compliance_agent_demo.ipynb)

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
| Agent orchestration | **LangGraph** | Boss agent's 3-node supervisor graph |
| Boss LLM | **Groq (GPT OSS 20B) via `langchain-groq`** | Free tier — intent selection + synthesis. Chosen over Llama 3.3 70B for likely higher free-tier rate limits (smaller model) and strong structured-output reliability; swappable for a paid provider later behind the same interface |
| Notebooks | **Jupyter + nbformat** | All runnable/demo scripts live as notebooks with markdown walkthroughs (`notebooks/`), not bare `.py` scripts — production code stays in `backend/` |
| SLM (planned) | **Ollama + qwen2.5:7b-instruct** | Briefing compression layer — not built yet |
| Frontend (planned) | **React (MERN) + Socket.io** | Not built yet |

---

## 6. Governance (Built-In, Not Bolted On)

Four pillars from the project spec, and what's actually implemented today:

| Pillar | Status |
|---|---|
| **Transparency** | Every `Finding` carries `confidence` + `severity`. Proxy-based tools (payment failure, refunds, margin) explicitly say so in their output rather than presenting estimates as facts. The synthesis LLM is explicitly prompted to record `Dissent`s rather than flatten disagreement into consensus — **demonstrated live** once Sentiment joined Finance: asked whether customer happiness is reflected in the numbers, the boss agent flagged Finance's improving margin against Sentiment's flat negative-review share and a 1.0/5-rated product as an explicit `Dissent`, not a smoothed-over summary. |
| **Traceability** | Every `Finding.source` names the exact tool that produced it. Every tool call — args, output, timing, success/failure — is captured in `ToolCallRecord`, automatically, via the base class. |
| **Human Oversight** | `BoardRecommendation.requires_human_approval` defaults to `True` (schema-level, not agent-level — can't be silently skipped). Now exercised end-to-end: every `BossAgent.run()` call produces a recommendation, never an auto-executed action. |
| **Accountability** | `GovernanceLog` schema exists to capture full session audit trails (query, agents invoked, findings, model versions, human decision). Not yet wired into a persistence layer (MongoDB — planned, not built). |

---

## 7. Environment Setup

Requires a `.env` file (gitignored, not committed) with:

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_KEY=your_kaggle_key
GROQ_API_KEY=your_groq_key            # free at console.groq.com — needed for the boss agent
```

```bash
pip install -r requirements.txt
jupyter notebook notebooks/            # open and run in order:
                                        #   01_data_ingestion.ipynb   (one-time: pull data + create Delta tables)
                                        #   02_finance_agent_demo.ipynb
                                        #   03_boss_agent_demo.ipynb
```

**Why notebooks for the runnable/demo layer, `.py` for everything importable**: `backend/`
holds code that other code imports — `FinanceAgent` is imported by `BossAgent`, which will
later be imported by a FastAPI route. That has to stay as clean, diffable, importable modules.
The `notebooks/` scripts are standalone demos nothing else imports, and benefit from inline
markdown explaining *why* each step happens plus visible output — exactly what notebooks are
good at and plain scripts aren't.

---

## 8. What's Next

Per the build order in `CLAUDE.md`:

1. ~~Repo scaffolding + Pydantic schemas + base `SpecialistAgent` class~~ ✅
2. ~~Databricks connection + Olist data ingestion into Delta tables~~ ✅
3. ~~Finance Agent end-to-end, all 7 tools~~ ✅
4. ~~Boss agent skeleton (LangGraph supervisor), wired to Finance Agent~~ ✅
5. ~~Remaining specialist agents~~ ✅ — Sentiment, Sales, Operations, Growth, Risk, Compliance/HR all built. **All 7 specialists complete.**
6. RAG layer (Databricks Vector Search) — real semantic search to replace the keyword-match proxies in `search_reviews()` (Sentiment) and `search_policy_docs()` (Compliance); institutional docs already built as part of Compliance ← next
7. Governance logging middleware persistence + MongoDB
8. MERN frontend + streaming + WebSocket agent status
9. Eval suite
10. Deployment containerization

Full project spec, including all 7 agents' complete tool lists and the schema reference, lives
in [`CLAUDE.md`](CLAUDE.md).
