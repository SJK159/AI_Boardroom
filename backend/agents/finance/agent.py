from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class FinanceAgent(SpecialistAgent):
    agent_type = AgentType.FINANCE

    def analyze(self, query: str) -> list[Finding]:
        findings = []

        margin = self._call_tool("calculate_margin_trend", tools.calculate_margin_trend, db=self.db)
        findings.append(Finding(
            claim=f"Contribution margin is {margin['trend_direction']} across the last {len(margin['monthly'])} months",
            source="calculate_margin_trend",
            confidence=0.75,
            supporting_data=margin,
            severity="warning" if margin["trend_direction"] == "declining" else "info",
        ))

        anomalies = self._call_tool("detect_revenue_anomalies", tools.detect_revenue_anomalies, db=self.db)
        findings.append(Finding(
            claim=f"{len(anomalies['anomalies'])} daily revenue anomalies detected out of {anomalies['days_analyzed']} days analyzed",
            source="detect_revenue_anomalies",
            confidence=0.85,
            supporting_data=anomalies,
            severity="warning" if anomalies["anomalies"] else "info",
        ))

        failure = self._call_tool("payment_failure_rate", tools.payment_failure_rate, db=self.db)
        findings.append(Finding(
            claim=f"Order failure rate (canceled/unavailable proxy) is {failure['failure_rate_pct']}% of {failure['total_orders']} orders",
            source="payment_failure_rate",
            confidence=0.6,
            supporting_data=failure,
            severity="warning" if failure["failure_rate_pct"] > 5 else "info",
        ))

        cogs = self._call_tool("calculate_cogs", tools.calculate_cogs, db=self.db)
        findings.append(Finding(
            claim="COGS cannot be calculated - source data has no unit cost field",
            source="calculate_cogs",
            confidence=1.0,
            supporting_data=cogs,
            severity="warning",
        ))

        forecast = self._call_tool("cash_flow_forecast", tools.cash_flow_forecast, db=self.db)
        next_period = forecast["forecast"][0]["projected_cash_in"] if forecast["forecast"] else None
        findings.append(Finding(
            claim=f"Next month's projected cash-in is {next_period}" if next_period is not None else "Insufficient history for a cash-flow forecast",
            source="cash_flow_forecast",
            confidence=0.55 if next_period is not None else 0.0,
            supporting_data=forecast,
            severity="info",
        ))

        refunds = self._call_tool("refund_impact_analysis", tools.refund_impact_analysis, db=self.db)
        findings.append(Finding(
            claim=f"{refunds['impact_pct']}% of total payment value ({refunds['at_risk_value']}) is at risk from canceled/unavailable orders",
            source="refund_impact_analysis",
            confidence=0.6,
            supporting_data=refunds,
            severity="warning" if refunds["impact_pct"] > 3 else "info",
        ))

        concentration = self._call_tool("revenue_concentration", tools.revenue_concentration, db=self.db)
        findings.append(Finding(
            claim=f"Revenue concentration is {concentration['concentration_level']} - top {concentration['top_n']} sellers hold {concentration['top_n_revenue_share_pct']}% of revenue (HHI {concentration['hhi']})",
            source="revenue_concentration",
            confidence=0.9,
            supporting_data=concentration,
            severity="critical" if concentration["concentration_level"] == "high" else "info",
        ))

        return findings
