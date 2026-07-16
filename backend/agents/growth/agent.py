from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class GrowthAgent(SpecialistAgent):
    agent_type = AgentType.GROWTH

    def analyze(self, query: str) -> list[Finding]:
        findings = []

        regional = self._call_tool("regional_sales_breakdown", tools.regional_sales_breakdown, db=self.db)
        top_state = regional["top_states"][0] if regional["top_states"] else None
        findings.append(Finding(
            claim=f"Top state by revenue is {top_state['state']} ({top_state['revenue']}) across {regional['states_analyzed']} states" if top_state else "No regional data available",
            source="regional_sales_breakdown",
            confidence=0.9,
            supporting_data=regional,
            severity="info",
        ))

        expansion = self._call_tool("market_expansion_signals", tools.market_expansion_signals, db=self.db)
        top_signal = expansion["signals"][0] if expansion["signals"] else None
        findings.append(Finding(
            claim=f"Strongest expansion signal is {top_signal['state']} at {top_signal['growth_pct']}% growth (recent 6mo vs prior 6mo)" if top_signal and top_signal["growth_pct"] is not None else "No clear expansion signal with an established baseline",
            source="market_expansion_signals",
            confidence=0.65,
            supporting_data=expansion,
            severity="info",
        ))

        cat_growth = self._call_tool("category_growth_rate", tools.category_growth_rate, db=self.db)
        top_cat = cat_growth["fastest_growing"][0] if cat_growth["fastest_growing"] else None
        findings.append(Finding(
            claim=f"Fastest-growing category is {top_cat['category']} at {top_cat['growth_pct']}% growth" if top_cat and top_cat["growth_pct"] is not None else "No category growth data with an established baseline",
            source="category_growth_rate",
            confidence=0.65,
            supporting_data=cat_growth,
            severity="info",
        ))

        acquisition = self._call_tool("customer_acquisition_trend", tools.customer_acquisition_trend, db=self.db)
        findings.append(Finding(
            claim=f"New customer acquisition is {acquisition['trend_direction']} across the last {len(acquisition['monthly'])} months",
            source="customer_acquisition_trend",
            confidence=0.85,
            supporting_data=acquisition,
            severity="warning" if acquisition["trend_direction"] == "declining" else "info",
        ))

        underperforming = self._call_tool("underperforming_region_diagnosis", tools.underperforming_region_diagnosis, db=self.db)
        worst = underperforming["underperforming_states"][0] if underperforming["underperforming_states"] else None
        findings.append(Finding(
            claim=f"{underperforming['states_declining']} states show declining revenue; worst is {worst['state']} at {worst['growth_pct']}%" if worst else "No states show a clear revenue decline",
            source="underperforming_region_diagnosis",
            confidence=0.65,
            supporting_data=underperforming,
            severity="warning" if worst else "info",
        ))

        onboarding = self._call_tool("new_seller_onboarding_rate", tools.new_seller_onboarding_rate, db=self.db)
        findings.append(Finding(
            claim=f"New seller onboarding is {onboarding['trend_direction']} across the last {len(onboarding['monthly'])} months",
            source="new_seller_onboarding_rate",
            confidence=0.85,
            supporting_data=onboarding,
            severity="warning" if onboarding["trend_direction"] == "declining" else "info",
        ))

        gaps = self._call_tool("product_gap_analysis", tools.product_gap_analysis, db=self.db)
        top_gap = gaps["potentially_undersupplied"][0] if gaps["potentially_undersupplied"] else None
        findings.append(Finding(
            claim=f"Most undersupplied category (by revenue-per-seller) is {top_gap['category']} ({top_gap['revenue_per_seller']}/seller, {top_gap['seller_count']} sellers)" if top_gap else "No category supply/demand data available",
            source="product_gap_analysis",
            confidence=0.4,
            supporting_data=gaps,
            severity="info",
        ))

        return findings
