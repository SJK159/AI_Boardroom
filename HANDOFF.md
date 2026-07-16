# HANDOFF — AI for the Boardroom

Purpose-built for switching machines (office Windows → home Mac) and for briefing a fresh
Claude session with zero prior context. This file is the map; [`README.md`](README.md) has
the narrative depth (why each decision was made, what bugs were found and fixed, verified
output examples). Read this first, then README for anything you need more detail on.

**Repo**: [github.com/SJK159/AI_Boardroom](https://github.com/SJK159/AI_Boardroom)
**Project spec**: [`CLAUDE.md`](CLAUDE.md) — the original brief this whole build follows

---

## 1. Resume-on-a-new-machine checklist (do this first, in order)

```bash
git clone https://github.com/SJK159/AI_Boardroom.git
cd AI_Boardroom

python3 -m venv venv && source venv/bin/activate     # recommended on Mac — avoids
                                                       # "externally-managed-environment" pip errors
pip install -r requirements.txt
```

**Recreate `.env`** at the repo root (gitignored, never committed — copy the actual values over
securely from the Windows machine, e.g. via a password manager, not plaintext chat/email):

```env
DATABRICKS_HOST=...
DATABRICKS_TOKEN=...
DATABRICKS_CATALOG=workspace        # optional, this is the default
DATABRICKS_SCHEMA=olist             # optional, this is the default
KAGGLE_USERNAME=...
KAGGLE_API_KEY=...
GROQ_API_KEY=...
MONGODB_URI=...
MONGODB_DB_NAME=ai_boardroom        # optional, this is the default
```

**Critical machine-specific step — rebuild the RAG vector indexes.** `backend/rag/.cache/` is
gitignored (regenerable derived data, same treatment as raw CSVs). The Mac will NOT have these
files after cloning. Run once:

```bash
jupyter notebook notebooks/10_rag_index_build.ipynb   # run all cells — takes ~5-6 min
```

Without this, `search_reviews()` and `search_policy_docs()` will raise a clear `RuntimeError`
pointing back to this notebook — that error is expected and correct on a fresh clone, not a bug.

**Do NOT re-run `notebooks/01_data_ingestion.ipynb`** unless you actually want to refresh the
Databricks tables. The Delta tables (`workspace.olist.*`) already exist in the cloud Databricks
workspace — that's server-side state, not local-machine state. `.env` pointing at the same
Databricks workspace is all that's needed to query them from the Mac. Re-running notebook 01
would just re-download the Kaggle CSVs and `CREATE OR REPLACE` the same tables (harmless but
unnecessary — ~10 min wasted for no benefit).

**Verify everything works**:

```bash
pytest -m "not llm and not mongo"   # fast tier, ~40s, no live LLM/Mongo calls — sanity check
                                     # that Databricks + RAG indexes are working
pytest                               # everything, ~4-5 min, exercises Groq + MongoDB too
```

If both pass, the environment is fully working on the new machine.

---

## 2. What this project is (one paragraph)

A multi-agent AI system that analyzes the Olist Brazilian E-Commerce dataset and delivers
decision support through a boardroom-style UX. Seven specialist agents (Finance, Sentiment,
Sales, Operations, Growth, Risk, Compliance/HR) each reason over their own domain with their
own tools; a boss agent dynamically selects which specialists are relevant to a query, runs
them in parallel, and synthesizes their findings into a recommendation — **explicitly
preserving disagreement** between agents rather than flattening it into false consensus. Full
concept/architecture explanation: README.md section 1-3.

---

## 3. What's built (complete)

Everything below is implemented, tested, and verified against live services.

| # | Component | Status | Where |
|---|---|---|---|
| 1 | Repo scaffolding, Pydantic schemas, base `SpecialistAgent` class | ✅ | `backend/schemas/`, `backend/agents/base_agent.py` |
| 2 | Databricks connection + Olist data ingestion (9 Delta tables) | ✅ | `notebooks/01_data_ingestion.ipynb`, `backend/db/` |
| 3-5 | All 7 specialist agents (49 tools total) | ✅ | `backend/agents/{finance,sentiment,sales,operations,growth,risk,compliance}/` |
| 4 | Boss agent — LangGraph 3-node supervisor graph | ✅ | `backend/agents/boss/` |
| 6 | RAG layer — local multilingual semantic search (2 collections) | ✅ | `backend/rag/`, `notebooks/10_rag_index_build.ipynb` |
| 7 | Governance logging — MongoDB Atlas session persistence | ✅ | `backend/governance/`, `notebooks/11_governance_demo.ipynb` |
| 9 | Eval suite — 27 pytest tests, 5 CLAUDE.md categories | ✅ | `tests/` (built ahead of step 8, at user's request) |

**Demo notebooks** (`notebooks/`), one per component, each runnable standalone:
`01_data_ingestion` → `02_finance_agent_demo` → `03_boss_agent_demo` →
`04_sentiment_agent_demo` → `05_sales_agent_demo` → `06_operations_agent_demo` →
`07_growth_agent_demo` → `08_risk_agent_demo` → `09_compliance_agent_demo` →
`10_rag_index_build` → `11_governance_demo`

**Backend structure** (full detail + rationale in README section 4):

```
backend/
├── schemas/       Pydantic contracts: Finding, ToolCallRecord, AgentBriefing,
│                  BoardRecommendation, Dissent, GovernanceLog
├── config/        settings.py — centralized env loading (pydantic-settings)
├── db/            databricks_client.py — the ONLY file that talks to the Databricks SDK
├── agents/
│   ├── base_agent.py   SpecialistAgent ABC — automatic per-tool-call logging
│   ├── finance/         7 tools
│   ├── sentiment/       7 tools (search_reviews uses the RAG layer)
│   ├── sales/            8 tools
│   ├── operations/       7 tools
│   ├── growth/            7 tools
│   ├── risk/               7 tools
│   ├── compliance/         7 tools + 3 fictional institutional docs (search_policy_docs uses the RAG layer)
│   └── boss/                LangGraph orchestrator, registry.py lists all 7 specialists
├── rag/            embedder.py, vector_index.py, review_index.py, policy_index.py
└── governance/     mongo_client.py, logger.py (GovernanceLogger wraps BossAgent)

tests/              27 pytest tests, markers: llm, mongo (unmarked = fast, no live calls)
```

---

## 4. What's remaining

Per CLAUDE.md's build order:

- **Step 8 — MERN frontend + streaming + WebSocket agent status.** Not started. Would need:
  a Node/Express + FastAPI bridge (CLAUDE.md section 8 stack) to expose `BossAgent`/
  `GovernanceLogger` over HTTP, a React frontend showing live "Finance Agent analyzing..."
  status via Socket.io, streamed synthesis token-by-token, dissent shown visibly, sources
  cited, confidence shown. This is genuinely the next open item.
- **Step 10 — Deployment containerization.** Not started. CLAUDE.md section 7 wants
  everything containerized and config-driven (env vars, not hardcoded infra) from day one —
  `Settings` (pydantic-settings) already reads everything from `.env`, so the config-driven
  part is already in good shape; what's missing is actual Dockerfiles/compose setup.

Everything else in the original 10-step build order is done.

### Smaller open threads (not blocking, worth knowing about)

- **Synthesis citation accuracy** varies slightly by model — GPT OSS 20B occasionally
  misattributes a tool to the wrong agent in a synthesis (observed once, documented in
  README section 4.7). Not fixed because it's inherent model variance, not a code bug — the
  Llama 3.3 70B / GPT OSS 120B upgrade path (one-line change in `backend/config/settings.py`,
  `boss_llm_model`) is the lever if this needs to improve.
- **`GovernanceLog` is written but never read back by anything except the demo notebook and
  eval suite** — there's no admin UI or query tool for browsing past sessions yet. Would
  naturally live in the step-8 frontend.
- Real Databricks Vector Search (vs. the local embeddings actually used) remains a possible
  future upgrade if data scale ever exceeds what fits in memory on one machine — see the
  tradeoff writeup in README section 4.12.

---

## 5. External services this project depends on

All four are **free tier**, chosen deliberately over paid alternatives (see "Key decisions"
below). A fresh machine needs working credentials for all four in `.env`:

| Service | Used for | Free tier | Signup |
|---|---|---|---|
| Databricks | Delta Lake tables, SQL warehouse | Depends on your workspace — this project uses an existing workspace, not a new signup | N/A (already provisioned) |
| Kaggle | Pulling the Olist dataset | Free, standard account | kaggle.com |
| Groq | Boss agent LLM (GPT OSS 20B) | Free tier, generous rate limits | console.groq.com |
| MongoDB Atlas | Governance session logs | M0 tier — free forever, 512MB, no trial expiry | mongodb.com/cloud/atlas |

---

## 6. Key architectural decisions (so a fresh Claude session doesn't relitigate these)

CLAUDE.md's stack table names specific paid/managed services in a few places. Each was
deliberately substituted for a free/local equivalent, matching the *capability* the spec
calls for without the cost — this was a recurring, explicit pattern across the build, not
scope-cutting:

| CLAUDE.md calls for | Actually used | Why |
|---|---|---|
| Hosted LLM for boss agent (implies Claude/GPT) | **Groq (GPT OSS 20B)** | Anthropic/OpenAI cost real money per token; Groq's free tier is fast and capable enough for orchestration + synthesis. `temperature=0` set for reproducible routing (see bug #4 below). |
| Databricks Vector Search | **Local embeddings** (`sentence-transformers`, `paraphrase-multilingual-MiniLM-L12-v2`, CPU only) | Vector Search is a separately billed managed endpoint; local embeddings deliver the same RAG capability at this dataset's scale (~41k reviews) for $0. Multilingual model means Portuguese reviews work with English queries — no translation step. |
| MongoDB/Mongoose (session storage) | **MongoDB Atlas M0 free tier** | Matches the spec exactly, just the free tier instead of a paid cluster. |
| Ollama + qwen2.5:7b-instruct (SLM briefing layer) | **Not built yet** | Out of scope so far — no substitution decision made here yet. |

All four are swappable later behind their existing interfaces if this ever needs to scale
past a single machine or a portfolio-project traffic level — none of this is a dead end.

**Other decisions worth knowing:**

- **Notebooks vs. `.py` scripts**: `backend/` is importable production code (`FinanceAgent` is
  imported by `BossAgent`, which will eventually be imported by a FastAPI route) — stays as
  clean `.py` modules. Everything runnable/demo-only lives in `notebooks/` with markdown
  walkthroughs. Established early, held consistently throughout.
- **Each specialist's `tools.py` is self-contained** — only imports `backend.db`, never
  imports another agent's module directly, even where CLAUDE.md describes genuine cross-agent
  data dependencies (e.g. Compliance's `cross_reference_sla_compliance` recomputes delivery
  metrics with the same methodology as Operations, rather than importing Operations' code).
- **"Flag data limitations, don't fabricate"** is the load-bearing honesty pattern across
  every agent — `calculate_cogs` (Finance), `carrier_performance_comparison` (Operations),
  `regulatory_exposure_check` (Risk), `product_gap_analysis` (Growth) all explicitly return
  "not available" + a reason rather than inventing a plausible-looking number. Follow this
  pattern for any new tool.

---

## 7. Real bugs found and fixed during this build (chronological)

Useful context so nobody re-discovers or accidentally reverts these. Full writeups with
before/after numbers are in the README sections cited.

1. **CSV multiline parsing corruption** (`order_reviews`, ~0.07% of rows) — review text with
   embedded newlines shifted columns for some rows. Fixed with `multiLine => true` in
   ingestion; defended in Sentiment's tools via `try_cast`. README 4.6.
2. **False "declining revenue" trend** (Sales) — Olist's data collection stops mid-Sept 2018;
   the naive "last 12 months" window included near-empty trailing months. Fixed by detecting
   and excluding trailing periods far below the window's median order count. README 4.7.
3. **Small-base growth-rate noise** (Growth) — a state with R$656 prior revenue showed
   "827.96% growth." Fixed with a minimum-prior-revenue filter before ranking. README 4.9.
4. **Non-deterministic boss-agent routing** (eval suite) — same query selected different
   specialists across runs because the Groq LLM had no `temperature` set. Fixed with
   `temperature=0` in `backend/agents/boss/graph.py`. README 4.14.
5. **Groq tool-call validation flakiness** (boss agent) — GPT OSS 20B intermittently returns
   a mismatched tool name during structured output. Fixed with a retry wrapper
   (`_invoke_with_retry`, up to 3 attempts) rather than trying to eliminate inherent model
   flakiness. README 4.9.
6. **Z-score over-flagging on skewed data** (Risk) — payment-value distribution has skewness
   ~9.15; `z >= 3.0` flagged 13x more "fraud signal" orders than a normal distribution would
   predict. Fixed by switching to a percentile threshold. README 4.10.
7. **Live `DatabricksClient` object leaking into every `AgentBriefing`** — `_call_tool()`
   logged the full `kwargs` dict (including the `db` connection object) as
   `ToolCallRecord.input_params`, present in every tool call since the first agent (Finance)
   was built. Invisible until MongoDB's strict BSON serialization caught it. Fixed by
   excluding `db` from logged params. README 4.13.
8. **`datetime.utcnow()` deprecation** across all 3 schema files with a timestamp default —
   cosmetic but a real future-Python-version break. Fixed alongside the eval suite build.

---

## 8. Mac-specific notes (vs. the Windows dev environment this was built on)

- No PowerShell — use Terminal (zsh/bash). Any command referencing `chcp 65001` (Windows
  console UTF-8 fix) doesn't apply on Mac.
- `pip install` may need a virtualenv on modern Mac Python (externally-managed-environment
  restriction) — see the venv step in section 1 above.
- `sentence-transformers` will download the embedding model fresh on first use
  (`~/.cache/huggingface` on Mac vs. the Windows equivalent) — one-time ~470MB download,
  happens automatically, no action needed.
- Jupyter notebook execution (`jupyter nbconvert --execute` or the Jupyter UI) works
  identically — no OS-specific notebook code in this project.

---

## 9. If you're a fresh Claude session picking this up

Read, in order: this file → `CLAUDE.md` (original spec) → `README.md` (full narrative/
verified output for whichever section is relevant to the current task). Don't re-verify
things marked ✅ above unless the user reports something's actually broken — they're tested
and were working as of the last commit. Check `git log --oneline` for the most recent commits
if you want the exact sequence of what happened when.
