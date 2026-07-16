"""Registry of specialist agents the boss agent can actually invoke.

Only agents listed here are offered to the selection LLM — the boss never
selects a specialist that isn't implemented yet. Adding a new specialist later
is one entry here, nothing else in the boss agent changes.
"""
from dataclasses import dataclass
from typing import Type

from backend.agents.base_agent import SpecialistAgent
from backend.agents.compliance import ComplianceAgent
from backend.agents.finance import FinanceAgent
from backend.agents.growth import GrowthAgent
from backend.agents.operations import OperationsAgent
from backend.agents.risk import RiskAgent
from backend.agents.sales import SalesAgent
from backend.agents.sentiment import SentimentAgent
from backend.schemas import AgentType


@dataclass
class SpecialistEntry:
    agent_class: Type[SpecialistAgent]
    description: str


AVAILABLE_SPECIALISTS: dict[AgentType, SpecialistEntry] = {
    AgentType.FINANCE: SpecialistEntry(
        agent_class=FinanceAgent,
        description=(
            "Margin and cost trends, revenue anomalies, payment failure/refund risk, "
            "cash-flow forecasting, and seller revenue concentration. Draws from orders, "
            "order_items, and order_payments."
        ),
    ),
    AgentType.RISK: SpecialistEntry(
        agent_class=RiskAgent,
        description=(
            "Cancellation trends, payment dispute risk, seller churn, customer revenue "
            "concentration, statistical fraud-screening signals, customer churn risk, and "
            "regulatory exposure (currently unavailable - no compliance data exists yet). "
            "Draws from orders, order_payments, order_items, and customers."
        ),
    ),
    AgentType.SALES: SpecialistEntry(
        agent_class=SalesAgent,
        description=(
            "Revenue trends, average order value, category and seller sales rankings, "
            "order fulfillment funnel drop-off, repeat purchase rate, seasonal patterns, "
            "and cross-sell category pairs. Draws from orders, order_items, customers, "
            "and products."
        ),
    ),
    AgentType.COMPLIANCE: SpecialistEntry(
        agent_class=ComplianceAgent,
        description=(
            "Institutional document fact retrieval: company registration status, HR policy "
            "coverage and gaps, seller contract clauses, contract renewal tracking, and "
            "seller SLA compliance (cross-references actual delivery performance against "
            "the contractual on-time threshold). Draws from local policy/contract documents "
            "plus orders, order_items, and sellers."
        ),
    ),
    AgentType.GROWTH: SpecialistEntry(
        agent_class=GrowthAgent,
        description=(
            "Regional sales breakdown, market expansion signals, category growth rates, "
            "customer acquisition and seller onboarding trends, underperforming-region "
            "diagnosis, and a supply/demand imbalance proxy for product gaps. Draws from "
            "orders, order_items, customers, sellers, and products."
        ),
    ),
    AgentType.OPERATIONS: SpecialistEntry(
        agent_class=OperationsAgent,
        description=(
            "Delivery delay and lateness rates, seller reliability (on-time delivery, not "
            "revenue), shipping cost efficiency, fulfillment stage bottlenecks, and delivery "
            "estimate accuracy. Draws from orders, order_items, sellers, and customers. No "
            "carrier-level data exists in the source."
        ),
    ),
    AgentType.SENTIMENT: SpecialistEntry(
        agent_class=SentimentAgent,
        description=(
            "Customer review analysis: sentiment by product/region, negative review trends, "
            "recurring complaint themes, and review-related correlations. Draws from "
            "order_reviews, products, and customers. Review text is Portuguese and untranslated."
        ),
    ),
}
