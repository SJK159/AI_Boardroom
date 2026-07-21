from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class RiskAgent(SpecialistAgent):
    agent_type = AgentType.RISK

    def analyze(self, query: str) -> list[Finding]:
        results = self._call_tools_parallel({
            "cancellation_trend": (tools.cancellation_trend, {"db": self.db}),
            "flag_payment_disputes": (tools.flag_payment_disputes, {"db": self.db}),
            "seller_churn_risk": (tools.seller_churn_risk, {"db": self.db}),
            "concentration_risk": (tools.concentration_risk, {"db": self.db}),
            "fraud_signal_detection": (tools.fraud_signal_detection, {"db": self.db}),
            "customer_churn_prediction": (tools.customer_churn_prediction, {"db": self.db}),
            "regulatory_exposure_check": (tools.regulatory_exposure_check, {"db": self.db}),
        })
        cancellations = results["cancellation_trend"]
        disputes = results["flag_payment_disputes"]
        seller_churn = results["seller_churn_risk"]
        concentration = results["concentration_risk"]
        fraud = results["fraud_signal_detection"]
        churn = results["customer_churn_prediction"]
        regulatory = results["regulatory_exposure_check"]

        findings = []

        findings.append(Finding(
            claim=f"Cancellation rate is {cancellations['trend_direction']} across the last {len(cancellations['monthly'])} months",
            source="cancellation_trend",
            confidence=0.85,
            supporting_data=cancellations,
            severity="warning" if cancellations["trend_direction"] == "worsening" else "info",
        ))

        findings.append(Finding(
            claim=f"{disputes['at_risk_orders']} canceled orders ({disputes['at_risk_value']} total) had payment already collected - dispute risk",
            source="flag_payment_disputes",
            confidence=0.55,
            supporting_data=disputes,
            severity="warning" if disputes["at_risk_orders"] > 0 else "info",
        ))

        findings.append(Finding(
            claim=f"{seller_churn['at_risk_pct']}% of sellers ({seller_churn['at_risk_count']} of {seller_churn['total_sellers']}) have gone inactive for {seller_churn['inactive_months_threshold']}+ months",
            source="seller_churn_risk",
            confidence=0.8,
            supporting_data=seller_churn,
            severity="warning" if seller_churn["at_risk_pct"] > 20 else "info",
        ))

        findings.append(Finding(
            claim=f"Customer revenue concentration is {concentration['concentration_level']} - top {concentration['top_n']} customers hold {concentration['top_n_revenue_share_pct']}% of revenue (HHI {concentration['hhi']})",
            source="concentration_risk",
            confidence=0.9,
            supporting_data=concentration,
            severity="critical" if concentration["concentration_level"] == "high" else "info",
        ))

        findings.append(Finding(
            claim=f"{fraud['flagged_count']} orders flagged as statistical payment-value outliers out of {fraud['orders_analyzed']} analyzed",
            source="fraud_signal_detection",
            confidence=0.45,
            supporting_data=fraud,
            severity="warning" if fraud["flagged_count"] > 0 else "info",
        ))

        findings.append(Finding(
            claim=f"{churn['at_risk_pct_of_repeat']}% of repeat customers ({churn['at_risk_count']} of {churn['repeat_customers_analyzed']}) are overdue for their next order",
            source="customer_churn_prediction",
            confidence=0.5,
            supporting_data=churn,
            severity="warning" if churn["at_risk_pct_of_repeat"] > 30 else "info",
        ))

        findings.append(Finding(
            claim="Regulatory exposure cannot be checked - no legal/compliance data exists in the source, and the Compliance/HR agent's institutional documents aren't built yet",
            source="regulatory_exposure_check",
            confidence=1.0,
            supporting_data=regulatory,
            severity="warning",
        ))

        return findings
