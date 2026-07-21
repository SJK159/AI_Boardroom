from concurrent.futures import ThreadPoolExecutor

from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class ComplianceAgent(SpecialistAgent):
    agent_type = AgentType.COMPLIANCE

    def analyze(self, query: str) -> list[Finding]:
        # 5 of 7 tools are independent of everything else and run in one parallel batch.
        # _top_seller_hint() is fetched concurrently alongside them (not itself a _call_tool,
        # so it can't go in the same dict, but there's no reason to make it wait its turn).
        # Only the 2 SLA-related tools genuinely depend on seller_hint's result, so they're
        # the one thing that has to wait for a prior round-trip to finish.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results_future = pool.submit(self._call_tools_parallel, {
                "search_policy_docs": (tools.search_policy_docs, {"db": self.db, "query": query}),
                "get_company_registration_info": (tools.get_company_registration_info, {"db": self.db}),
                "policy_gap_analysis": (tools.policy_gap_analysis, {"db": self.db}),
                "check_policy_compliance": (tools.check_policy_compliance, {"db": self.db, "topic": "Leave Policy"}),
                "contract_expiry_tracker": (tools.contract_expiry_tracker, {"db": self.db}),
            })
            seller_hint_future = pool.submit(self._top_seller_hint)
            results = results_future.result()
            seller_hint = seller_hint_future.result()

        sla_results = self._call_tools_parallel({
            "check_contract_clause": (tools.check_contract_clause, {"db": self.db, "vendor": seller_hint, "topic": "Service Level Agreement"}),
            "cross_reference_sla_compliance": (tools.cross_reference_sla_compliance, {"db": self.db, "vendor": seller_hint}),
        })

        search = results["search_policy_docs"]
        registration = results["get_company_registration_info"]
        gaps = results["policy_gap_analysis"]
        leave_policy = results["check_policy_compliance"]
        expiry = results["contract_expiry_tracker"]
        sla_clause = sla_results["check_contract_clause"]
        sla = sla_results["cross_reference_sla_compliance"]

        findings = []

        findings.append(Finding(
            claim=f"Semantic search for '{query}' matched {search['match_count']} document sections above the relevance threshold",
            source="search_policy_docs",
            confidence=0.7,
            supporting_data=search,
            severity="info",
        ))

        status = registration["sections"].get("Registered Status", "unknown")
        findings.append(Finding(
            claim=f"Company registration status: {status}",
            source="get_company_registration_info",
            confidence=1.0,
            supporting_data=registration,
            severity="info",
        ))

        findings.append(Finding(
            claim=f"HR policy covers {gaps['coverage_pct']}% of expected topics; missing: {', '.join(gaps['missing_topics']) if gaps['missing_topics'] else 'none'}",
            source="policy_gap_analysis",
            confidence=0.9,
            supporting_data=gaps,
            severity="warning" if gaps["missing_topics"] else "info",
        ))

        findings.append(Finding(
            claim=f"Seller contract SLA clause found: {sla_clause['matched_section']}" if sla_clause["matched_section"] else "No matching contract clause found for 'Service Level Agreement'",
            source="check_contract_clause",
            confidence=0.85,
            supporting_data=sla_clause,
            severity="info",
        ))

        findings.append(Finding(
            claim=f"HR policy on 'Leave Policy' is {'documented' if leave_policy['covered'] else 'NOT documented'}",
            source="check_policy_compliance",
            confidence=0.9,
            supporting_data=leave_policy,
            severity="info" if leave_policy["covered"] else "warning",
        ))

        findings.append(Finding(
            claim=f"{expiry['renewing_soon_count']} of {expiry['total_sellers']} seller agreements renew within {expiry['renewal_window_days']} days",
            source="contract_expiry_tracker",
            confidence=0.75,
            supporting_data=expiry,
            severity="warning" if expiry["renewing_soon_count"] > 0 else "info",
        ))

        if sla["found"]:
            findings.append(Finding(
                claim=f"Seller {sla['vendor']} is {'compliant' if sla['compliant'] else 'in breach'} with the {sla['contractual_threshold_pct']}% SLA at {sla['actual_on_time_pct']}% actual on-time delivery",
                source="cross_reference_sla_compliance",
                confidence=0.85,
                supporting_data=sla,
                severity="critical" if sla["compliant"] is False else "info",
            ))

        return findings

    def _top_seller_hint(self) -> str:
        """Picks a real seller_id to demo cross_reference_sla_compliance against.

        A real deployment would take `vendor` as a query parameter (e.g. the boss agent
        extracting a seller name/ID from the user's question) - this hint keeps the tool
        exercised end-to-end without requiring query parsing that's out of scope here.
        """
        rows = self.db.query(f"""
            SELECT seller_id, COUNT(*) AS n
            FROM {self.db.table('order_items')}
            GROUP BY seller_id
            ORDER BY n DESC
            LIMIT 1
        """)
        return rows[0]["seller_id"] if rows else ""
