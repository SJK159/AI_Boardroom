# AI for the Boardroom — Project Context

This file is persistent context for Claude Code. Read this in full before starting any work.
This project is being built **in increments** — do not attempt to build everything in one pass.
Wait for explicit direction on which piece to build next, unless told otherwise.

---

## 1. What This Project Is

A multi-agent AI system that analyzes company data and delivers decision support through a
**boardroom-style UX**. Specialist AI agents act as board personas (Finance, Sales, Sentiment,
Operations, Growth, Risk, Compliance/HR), each equipped with domain-specific tools. A **boss
agent** dynamically decides which specialists are relevant to a given query, invokes them
(in parallel where possible), and synthesizes their findings into a final recommendation —
**explicitly preserving disagreement** between agents rather than flattening it into false
consensus.

This is a portfolio/agency-demo project. Full scope is intended — **all 7 agents, full tool
sets (no v1 trimming)**. The owner (Saad) is a final-year CS student with strong hands-on
experience in LangGraph, agentic AI, and multi-agent orchestration (built a LangGraph agent
with Mongoose-backed tools, checkpointing, streaming, and config-injection security as part of
LUMS coursework) — so build assuming solid technical judgment, not a beginner audience. Prefer
being asked before major architectural pivots, but don't over-explain fundamentals.

**Origin note**: the concept started from tree-of-thought prompting (branches = board members
debating). This evolved into the multi-agent + boss-agent pattern below because it's more
practically buildable, more traceable (tool-backed conclusions vs. pure reasoning), and easier
to govern. **ToT is deliberately deferred** — it may be reintroduced later as an *internal*
reasoning tool inside the boss agent specifically, for resolving sharply conflicting specialist
findings. Do not build ToT into v1 unless explicitly asked.

---

## 2. Data Sources

**Primary dataset: Olist Brazilian E-Commerce dataset** (public, Kaggle). ~100k orders
(2016–2018) across relational tables: orders, order_items, products, customers, sellers,
payments, and reviews (free-text). Gives both structured transactional data and unstructured
review text.

- Review text is mostly **Portuguese** — needs a translation step or a multilingual embedding
  model in the RAG pipeline. Decide/flag this explicitly when building the Sentiment Agent.

**Supplementary institutional documents** (simulated — do NOT use real company data):
- Company Registration Certificate — incorporation facts, jurisdiction, status
- HR Policy Document — employment terms, leave policy, code of conduct, compensation, termination
- Vendor/Seller Contract — scope, payment terms, SLA clauses, termination, liability

These are built as fictional documents for a wrapper entity ("Olist Inc.") once we're hands-on
with the data — **not built in advance**, so don't fabricate content for these until asked.
They require a different RAG treatment than reviews: smaller chunks, section/clause-based
chunking (not fixed token windows), metadata tagging (`doc_type: policy|registration|contract`),
and a **separate Vector Search collection** from the reviews index — precision fact-retrieval,
not pattern retrieval.

---

## 3. System Architecture

Five layers, in order:

1. **RAG (Databricks)** — Delta Lake tables for structured data (orders, payments, etc.).
   Databricks Vector Search for embedded reviews and policy/contract docs (separate,
   metadata-tagged collections). Unity Catalog for per-tenant governance/isolation.
2. **NLP/DL (extraction layer)** — batch processing: entity extraction, sentiment scoring,
   anomaly detection, document classification. Writes enriched output back to Delta tables.
   Feeds both RAG and the specialist agents.
3. **SLM (briefing layer)** — Ollama + `qwen2.5:7b-instruct` compresses NLP/DL output + RAG
   retrieval into a validated briefing packet per agent per query. Kept lightweight/deployable
   since it runs frequently.
4. **Agents (specialists + boss)** — specialist agents reason over their briefing packet using
   their own tools (Delta SQL queries, RAG retrieval, computed metrics). Boss agent (stronger
   hosted LLM) orchestrates and synthesizes. Built on **LangGraph's supervisor/multi-agent
   pattern**.
5. **UX layer** — React (MERN stack), WebSocket-driven live agent status ("Finance Agent
   analyzing...") + streamed boss-agent synthesis token-by-token. Dissent visible, sources
   cited, confidence shown.

**Query-time flow:**
```
User query
  -> Boss Agent parses intent
  -> Boss Agent selects relevant specialist agents (dynamic, not fixed — this is the
     "which board members are needed for this decision" logic)
  -> Selected agents run in PARALLEL where possible
  -> Each agent invokes its own tools
  -> Each agent produces a validated AgentBriefing (see schema below)
  -> Boss Agent collects ALL briefings
  -> Boss Agent synthesizes into a board memo, detecting and preserving disagreement,
     citing sources
  -> Streamed to UX layer
```

**Core design rule**: the SLM and the boss LLM never talk to each other directly. Everything
funnels through the validated briefing packet. This keeps every layer independently swappable
and caps expensive LLM calls at once per board session (not once per data chunk).

---

## 4. Agent Roster & Tools (Full Scope — No Trimming)

Seven specialist agents + one boss agent. All tools below are in scope for full build-out.

### Finance Agent — margin, cost, cash flow
- `calculate_margin_trend()`
- `detect_revenue_anomalies()`
- `payment_failure_rate()`
- `calculate_cogs()`
- `cash_flow_forecast()`
- `refund_impact_analysis()`
- `revenue_concentration()`

### Sales Agent — revenue behavior, conversion
- `query_revenue_by_period()`
- `calculate_aov()`
- `sales_by_category()`
- `seller_sales_ranking()`
- `conversion_funnel_analysis()`
- `repeat_purchase_rate()`
- `seasonal_sales_pattern()`
- `cross_sell_opportunities()`

### Customer Sentiment Agent — review & feedback analysis
- `search_reviews(query)` — RAG
- `sentiment_score_by_product()`
- `flag_negative_trend()`
- `extract_common_complaints()`
- `review_response_time_correlation()`
- `sentiment_by_region()`
- `photo_review_analysis()`

### Operations/Logistics Agent — delivery & fulfillment
- `calculate_delivery_delay()`
- `seller_performance_score()`
- `flag_late_shipments()`
- `shipping_cost_analysis()`
- `carrier_performance_comparison()`
- `fulfillment_bottleneck_detection()`
- `estimated_vs_actual_delivery_accuracy()`

### Growth Agent — expansion & opportunity signals
- `regional_sales_breakdown()`
- `market_expansion_signals()`
- `category_growth_rate()`
- `customer_acquisition_trend()`
- `underperforming_region_diagnosis()`
- `new_seller_onboarding_rate()`
- `product_gap_analysis()` — **weakest fit for Olist data; may need scoping down or
  supplementary data. Flag this when implementing rather than forcing a fake signal.**

### Risk Agent — downside/threat detection
- `cancellation_trend()`
- `flag_payment_disputes()`
- `seller_churn_risk()`
- `concentration_risk()`
- `fraud_signal_detection()`
- `customer_churn_prediction()`
- `regulatory_exposure_check()`

### Compliance/HR Agent — fact retrieval from institutional docs
- `search_policy_docs(query)` — RAG, scoped to policy/contract collection only
- `get_company_registration_info()`
- `check_contract_clause(vendor, topic)`
- `check_policy_compliance(topic)`
- `contract_expiry_tracker()`
- `policy_gap_analysis(topic)`
- `cross_reference_sla_compliance(vendor)` — **cross-agent tool**: pulls Operations delivery
  data and compares against contractual SLA terms. No single agent can answer this alone —
  build this carefully as it's the strongest demo of why the multi-agent design earns its
  complexity.

**Known intentional overlaps** (do not "fix" these — they're deliberate, different lenses on
the same entity):
- `payment_failure_rate()` (Finance) vs `flag_payment_disputes()` (Risk) — financial impact vs.
  threat pattern.
- `shipping_cost_analysis()` (Ops) vs Finance's cost tools — logistics efficiency vs. margin
  impact.
- `seller_sales_ranking()` (Sales) vs `seller_performance_score()` (Ops) — volume vs.
  reliability. The boss agent cross-referencing these ("top seller by volume is also your
  worst on delivery") is a good demo moment.

### Boss Agent — orchestration & synthesis
- Not a "specialist" — no domain tools of its own beyond `request_clarification()` (optional,
  loops back to a specialist for more detail).
- Responsibilities: parse query intent → select relevant specialists → invoke in parallel →
  collect briefings → detect disagreement → synthesize board memo → (future) internal ToT pass
  on sharp conflicts before finalizing.
- **Model choice**: stronger hosted LLM (e.g. Claude/GPT via API), NOT the local SLM. Runs
  once per session, so it's the right place to spend compute/cost — specialist agents + SLM
  stay lightweight since they run per-query, per-agent.

---

## 5. Schemas (Validated — Pydantic + Zod)

Everything flowing between agents is validated. This is Option C from planning: validated
schemas from day one, not retrofitted. Reference implementation (already scaffolded, may need
refinement during build):

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime
from enum import Enum

class AgentType(str, Enum):
    FINANCE = "Finance"
    SALES = "Sales"
    SENTIMENT = "Sentiment"
    OPERATIONS = "Operations"
    GROWTH = "Growth"
    RISK = "Risk"
    COMPLIANCE = "Compliance"

class Finding(BaseModel):
    claim: str
    source: str                      # tool name or doc citation
    confidence: float = Field(ge=0, le=1)
    supporting_data: dict = Field(default_factory=dict)
    severity: Optional[Literal["info", "warning", "critical"]] = "info"

class ToolCallRecord(BaseModel):
    tool_name: str
    input_params: dict = Field(default_factory=dict)
    output_summary: str
    execution_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

class AgentBriefing(BaseModel):
    agent: AgentType
    findings: list[Finding] = Field(min_length=1)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[float] = None

class Dissent(BaseModel):
    agents_involved: list[AgentType] = Field(min_length=2)
    topic: str
    summary: str
    resolution: Optional[str] = None

class BoardRecommendation(BaseModel):
    query: str
    agents_invoked: list[AgentType]
    briefings: list[AgentBriefing]
    synthesis: str
    dissents: list[Dissent] = Field(default_factory=list)
    confidence_overall: float = Field(ge=0, le=1)
    action_items: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True   # governance: recommendation, not auto-action
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class GovernanceLog(BaseModel):
    session_id: str
    user_query: str
    recommendation: BoardRecommendation
    model_versions: dict = Field(default_factory=dict)
    human_decision: Optional[Literal["accepted", "rejected", "modified"]] = None
    human_notes: Optional[str] = None
    total_execution_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

Mirror these as Zod schemas / TypeScript types on the MERN side so frontend and backend agree
on shape without duplicated logic (generate from Pydantic where practical, hand-mirror
otherwise since they're small).

---

## 6. AI Governance Layer

Governance is architectural, not bolted on. Four pillars and their implementation:

| Pillar | Implementation |
|---|---|
| Transparency | Dissent between agents shown explicitly in the board memo, never flattened to consensus. |
| Traceability/Provenance | Every claim cites its source (tool output or doc chunk) — enforced in the UX layer, not optional. |
| Human Oversight | Board produces a *recommendation* only. Explicit accept/reject/modify required before anything is "decided." |
| Accountability | Every session logged: query, agents invoked, tool calls, findings, synthesis, model versions, human decision. |

Further implementation to build in (not afterthoughts):
- Logging middleware wrapping every agent call — consistent capture, not per-agent self-report.
- Confidence scoring per finding, surfaced distinctly in UX (high/medium/low).
- Explicit scope boundaries on the Compliance/HR agent around anything resembling employment
  decisions (EU AI Act high-risk category — relevant given Saad's Txend AI governance work).
- Model/version logging per session for explainability across model swaps.

This roughly aligns with EU AI Act transparency expectations for high-risk systems, even though
this tool likely wouldn't be classified high-risk itself. Build to that standard anyway — it's
a genuine differentiator and directly relevant to ongoing related work.

---

## 7. Deployment Strategy

Target: **international SMB clients**, general-purpose (not niched to a regulated industry).

- **Hosted SaaS with regional data residency** — **default for v1**. Zero client-side setup,
  familiar SaaS trust model, fastest to build/iterate.
- **BYOC (client's own cloud account)** — future tier for trust-sensitive clients. Data never
  touches your infra.
- **Local/on-device (SLM only)** — future tier, strongest trust story, real cross-hardware
  support burden.

**Build rule**: containerize everything and keep config-driven (env vars/secrets, not
hardcoded infra assumptions) from day one, so BYOC/on-prem tiers are a future deploy target,
not a rebuild.

---

## 8. Technology Stack

| Layer | Technology |
|---|---|
| Data / RAG | Databricks — Delta Lake + Vector Search, Unity Catalog |
| SLM (specialist agents) | Ollama + `qwen2.5:7b-instruct` |
| LLM (boss agent) | Hosted API — stronger model, synthesis-only calls |
| Agent orchestration | LangGraph — supervisor/multi-agent pattern |
| Schema validation | Pydantic (Python) + Zod (TypeScript) |
| Backend | Node/Express + FastAPI bridge to the Python agent layer |
| Frontend | React (MERN), streamed output, Socket.io for live agent status |
| Session storage | MongoDB/Mongoose — governance logs, session records |
| Document parsing | PyMuPDF + multilingual embedding model (Portuguese reviews) |

---

## 9. Evaluation Strategy

Not a single demo query — a regression eval suite, run on every prompt/model/schema change:

- **Factual accuracy** — agent tool output vs. ground-truth calculation on raw data (pass/fail).
- **Retrieval relevance** — RAG results vs. hand-labeled relevance set (precision/recall).
- **Synthesis quality** — does the boss agent preserve known conflicting findings? LLM-as-judge
  rubric (qualitative, not pass/fail).
- **Governance completeness** — every session log has a full audit trail (pass/fail, structural).
- **Robustness** — ambiguous queries, single-agent-only queries, no-relevant-data queries;
  should fail gracefully, not hallucinate.

---

## 10. How We Work — Build Process

- **This is an incremental build.** Do not attempt to scaffold the entire system in one session.
  Confirm scope with Saad before starting a new major component.
- **Full scope, no shortcuts.** All 7 agents and their full tool sets are in scope for
  completion — earlier discussion explicitly rejected trimming to a v1 subset. Pace via
  increments, not via cutting scope.
- **Validate before trusting.** Every agent output must pass the Pydantic schema before being
  treated as valid. Fail loud, not silent.
- **Governance is not optional polish** — logging, source citation, and human-approval gating
  should be present from the first working agent, not added at the end.
- **Simplicity in implementation preferred** over cleverness — this was an explicit design
  goal from planning (agents + tools over hand-rolled tree-of-thought orchestration).
- **No `goto` statements** (general coding preference).
- For coding *assignments* (not this project) Saad prefers guidance over full solution code —
  not applicable here since this is his own project, full implementation is expected.

### Suggested build order (confirm before proceeding at each step)
1. Repo scaffolding + Pydantic schemas + base `SpecialistAgent` class (already started —
   check `/schemas`, `/agents/base_agent.py` if resuming from prior work)
2. Databricks connection + Olist data ingestion into Delta tables
3. One full specialist agent end-to-end (Finance recommended — clearest data mapping) with all
   7 tools, tested against real Delta-table queries
4. Boss agent skeleton (LangGraph supervisor) wired to the one working specialist
5. Remaining 6 specialist agents, one at a time
6. RAG layer (Vector Search) for Sentiment Agent + simulated institutional docs for Compliance
   Agent
7. Governance logging middleware + MongoDB session storage
8. MERN frontend + streaming + WebSocket agent status
9. Eval suite
10. Deployment containerization

Start wherever Saad directs — this order is a suggestion, not a requirement.
