from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class SalesAgent(SpecialistAgent):
    agent_type = AgentType.SALES

    def analyze(self, query: str) -> list[Finding]:
        results = self._call_tools_parallel({
            "query_revenue_by_period": (tools.query_revenue_by_period, {"db": self.db}),
            "calculate_aov": (tools.calculate_aov, {"db": self.db}),
            "sales_by_category": (tools.sales_by_category, {"db": self.db}),
            "seller_sales_ranking": (tools.seller_sales_ranking, {"db": self.db}),
            "conversion_funnel_analysis": (tools.conversion_funnel_analysis, {"db": self.db}),
            "repeat_purchase_rate": (tools.repeat_purchase_rate, {"db": self.db}),
            "seasonal_sales_pattern": (tools.seasonal_sales_pattern, {"db": self.db}),
            "cross_sell_opportunities": (tools.cross_sell_opportunities, {"db": self.db}),
        })
        revenue = results["query_revenue_by_period"]
        aov = results["calculate_aov"]
        by_category = results["sales_by_category"]
        sellers = results["seller_sales_ranking"]
        funnel = results["conversion_funnel_analysis"]
        repeat = results["repeat_purchase_rate"]
        seasonal = results["seasonal_sales_pattern"]
        cross_sell = results["cross_sell_opportunities"]

        findings = []

        findings.append(Finding(
            claim=f"Revenue is {revenue['trend_direction']} across the last {len(revenue['periods'])} {revenue['period_unit']}s",
            source="query_revenue_by_period",
            confidence=0.85,
            supporting_data=revenue,
            severity="warning" if revenue["trend_direction"] == "declining" else "info",
        ))

        findings.append(Finding(
            claim=f"Overall average order value is {aov['overall_aov']}",
            source="calculate_aov",
            confidence=0.9,
            supporting_data=aov,
            severity="info",
        ))

        top_cat = by_category["top_categories"][0] if by_category["top_categories"] else None
        findings.append(Finding(
            claim=f"Top category by revenue is {top_cat['category']} ({top_cat['revenue']})" if top_cat else "No category data available",
            source="sales_by_category",
            confidence=0.9,
            supporting_data=by_category,
            severity="info",
        ))

        top_seller = sellers["top_sellers"][0] if sellers["top_sellers"] else None
        findings.append(Finding(
            claim=f"Top seller by revenue is {top_seller['seller_id']} ({top_seller['revenue']})" if top_seller else "No seller data available",
            source="seller_sales_ranking",
            confidence=0.9,
            supporting_data=sellers,
            severity="info",
        ))

        findings.append(Finding(
            claim=f"{funnel['drop_off_pct']}% of orders drop off (canceled/unavailable) out of {funnel['total_orders']} total",
            source="conversion_funnel_analysis",
            confidence=0.65,
            supporting_data=funnel,
            severity="warning" if funnel["drop_off_pct"] > 5 else "info",
        ))

        findings.append(Finding(
            claim=f"Repeat purchase rate is {repeat['repeat_rate_pct']}% ({repeat['repeat_customers']} of {repeat['total_customers']} customers)",
            source="repeat_purchase_rate",
            confidence=0.9,
            supporting_data=repeat,
            severity="warning" if repeat["repeat_rate_pct"] < 5 else "info",
        ))

        findings.append(Finding(
            claim=f"Peak sales month is {seasonal['peak_month']}, trough is {seasonal['trough_month']}" if seasonal["peak_month"] else "Insufficient data for seasonal pattern",
            source="seasonal_sales_pattern",
            confidence=0.6,
            supporting_data=seasonal,
            severity="info",
        ))

        top_pair = cross_sell["category_pairs"][0] if cross_sell["category_pairs"] else None
        findings.append(Finding(
            claim=f"Strongest cross-sell pair is {top_pair['category_a']} + {top_pair['category_b']} ({top_pair['orders_together']} orders)" if top_pair else "No category pairs meet the co-occurrence threshold",
            source="cross_sell_opportunities",
            confidence=0.7,
            supporting_data=cross_sell,
            severity="info",
        ))

        return findings
