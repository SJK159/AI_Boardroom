"""Registry of specialist agents the boss agent can actually invoke.

Only agents listed here are offered to the selection LLM — the boss never
selects a specialist that isn't implemented yet. Adding a new specialist later
is one entry here, nothing else in the boss agent changes.
"""
from dataclasses import dataclass
from typing import Type

from backend.agents.base_agent import SpecialistAgent
from backend.agents.finance import FinanceAgent
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
    AgentType.SALES: SpecialistEntry(
        agent_class=SalesAgent,
        description=(
            "Revenue trends, average order value, category and seller sales rankings, "
            "order fulfillment funnel drop-off, repeat purchase rate, seasonal patterns, "
            "and cross-sell category pairs. Draws from orders, order_items, customers, "
            "and products."
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
