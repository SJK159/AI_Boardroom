from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class OperationsAgent(SpecialistAgent):
    agent_type = AgentType.OPERATIONS

    def analyze(self, query: str) -> list[Finding]:
        findings = []

        delay = self._call_tool("calculate_delivery_delay", tools.calculate_delivery_delay, db=self.db)
        findings.append(Finding(
            claim=f"{delay['late_pct']}% of deliveries arrive late (avg delay {delay['avg_delay_days']} days) out of {delay['total_delivered']} delivered orders",
            source="calculate_delivery_delay",
            confidence=0.9,
            supporting_data=delay,
            severity="warning" if delay["late_pct"] > 10 else "info",
        ))

        seller_perf = self._call_tool("seller_performance_score", tools.seller_performance_score, db=self.db)
        worst = seller_perf["least_reliable"][0] if seller_perf["least_reliable"] else None
        findings.append(Finding(
            claim=f"Least reliable seller (min {seller_perf['min_orders_threshold']} orders) is {worst['seller_id']} at {worst['on_time_pct']}% on-time" if worst else "No sellers meet the minimum order threshold",
            source="seller_performance_score",
            confidence=0.85,
            supporting_data=seller_perf,
            severity="warning" if worst and float(worst["on_time_pct"]) < 70 else "info",
        ))

        late_shipments = self._call_tool("flag_late_shipments", tools.flag_late_shipments, db=self.db)
        findings.append(Finding(
            claim=f"{late_shipments['severe_late_pct']}% of deliveries are severely late (>{late_shipments['severe_threshold_days']} days), {late_shipments['mild_late_pct']}% mildly late",
            source="flag_late_shipments",
            confidence=0.9,
            supporting_data=late_shipments,
            severity="critical" if late_shipments["severe_late_pct"] > 5 else "warning" if late_shipments["severe_late_pct"] > 1 else "info",
        ))

        shipping_cost = self._call_tool("shipping_cost_analysis", tools.shipping_cost_analysis, db=self.db)
        by_type = {r["shipment_type"]: r for r in shipping_cost["by_shipment_type"]}
        findings.append(Finding(
            claim=f"Interstate shipments average {by_type.get('interstate', {}).get('avg_freight', 'N/A')} freight vs {by_type.get('intrastate', {}).get('avg_freight', 'N/A')} for intrastate",
            source="shipping_cost_analysis",
            confidence=0.85,
            supporting_data=shipping_cost,
            severity="info",
        ))

        carrier = self._call_tool("carrier_performance_comparison", tools.carrier_performance_comparison, db=self.db)
        findings.append(Finding(
            claim="Carrier performance cannot be compared - no carrier identifier exists in the source data",
            source="carrier_performance_comparison",
            confidence=1.0,
            supporting_data=carrier,
            severity="warning",
        ))

        bottleneck = self._call_tool("fulfillment_bottleneck_detection", tools.fulfillment_bottleneck_detection, db=self.db)
        findings.append(Finding(
            claim=f"The slowest fulfillment stage is '{bottleneck['bottleneck_stage']}' at {bottleneck['stage_durations_days'][bottleneck['bottleneck_stage']]} days on average",
            source="fulfillment_bottleneck_detection",
            confidence=0.85,
            supporting_data=bottleneck,
            severity="info",
        ))

        accuracy = self._call_tool("estimated_vs_actual_delivery_accuracy", tools.estimated_vs_actual_delivery_accuracy, db=self.db)
        findings.append(Finding(
            claim=f"Delivery estimate accuracy: mean absolute error {accuracy['mae_days']} days, bias {accuracy['bias_days']} days ({'late-leaning' if accuracy['bias_days'] > 0 else 'early-leaning'})",
            source="estimated_vs_actual_delivery_accuracy",
            confidence=0.9,
            supporting_data=accuracy,
            severity="info",
        ))

        return findings
