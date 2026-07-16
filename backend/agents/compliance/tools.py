"""Compliance/HR Agent tools.

Data reality notes (flagged explicitly per tool output, not hidden):

1. search_policy_docs() is a keyword-match proxy over the local documents, same treatment as
   Sentiment's search_reviews() and for the same reason - real Vector Search (with doc_type
   metadata tagging, a separate collection from reviews per CLAUDE.md section 2) is scoped to
   build order step 6, not built yet.
2. check_contract_clause() accepts a `vendor` (seller_id) parameter, but all sellers currently
   operate under ONE standard agreement template - there are no per-vendor bespoke contracts in
   this dataset. The parameter validates the seller exists and is accepted for interface
   consistency with a future state where individual addenda might exist.
3. contract_expiry_tracker() reads the contract's renewal cadence ("annual basis") as a fixed
   365-day period rather than parsing natural-language duration text - a deliberate
   simplification. A production system would store contract terms as structured metadata
   alongside the prose, not re-parse English date language on every call.
4. cross_reference_sla_compliance() computes seller on-time delivery rate directly against the
   Delta tables using the SAME methodology as Operations' calculate_delivery_delay/
   seller_performance_score (not by importing Operations' module - this codebase keeps each
   specialist's tools.py self-contained, only backend.db is shared). It's a "cross-agent" tool
   in the data sense CLAUDE.md describes: it reaches into Operations' domain data to answer a
   question no single agent's own lens could answer alone.
5. policy_gap_analysis() only reports which expected HR topics are documented vs. missing. It
   does not recommend or make employment decisions - CLAUDE.md section 6 explicitly scopes this
   agent away from anything resembling employment decisions (EU AI Act high-risk category).
"""
import re
from datetime import datetime, timedelta

from backend.db import DatabricksClient

from .document_loader import find_section, load_all_documents, load_document

EXPECTED_HR_TOPICS = [
    "Employment Classifications",
    "Leave Policy",
    "Code of Conduct",
    "Compensation & Performance Reviews",
    "Termination Policy",
    "Data Privacy Policy",
    "Anti-Discrimination & Equal Opportunity",
    "Remote Work Policy",
    "Grievance Procedure",
]


def _escape(s: str) -> str:
    return s.replace("'", "''")[:200]


def search_policy_docs(db: DatabricksClient, query: str, limit: int = 5) -> dict:
    """Keyword search across all institutional documents - a proxy for real semantic search."""
    query_lower = query.lower()
    matches = []
    for doc in load_all_documents():
        for header, body in doc["sections"].items():
            if query_lower in header.lower() or query_lower in body.lower():
                matches.append({
                    "doc_type": doc["doc_type"],
                    "document": doc["title"],
                    "section": header,
                    "excerpt": body[:300],
                })
    return {
        "query": query,
        "matches": matches[:limit],
        "match_count": len(matches),
        "note": "literal keyword match across institutional documents, not semantic/embedding search",
    }


def get_company_registration_info(db: DatabricksClient) -> dict:
    """Structured fields from the Company Registration Certificate - precision fact retrieval."""
    doc = load_document("company_registration.md")
    return {"sections": doc["sections"], "source_document": doc["title"]}


def check_contract_clause(db: DatabricksClient, vendor: str, topic: str) -> dict:
    """Look up a specific clause in the standard seller agreement for a given seller."""
    safe_vendor = _escape(vendor)
    seller_check = db.query(f"SELECT COUNT(*) AS n FROM {db.table('sellers')} WHERE seller_id = '{safe_vendor}'")
    seller_exists = int(seller_check[0]["n"]) > 0

    doc = load_document("vendor_contract.md")
    matched_section = find_section(doc, topic)

    return {
        "vendor": vendor,
        "vendor_found_in_sellers_table": seller_exists,
        "topic": topic,
        "matched_section": matched_section,
        "clause_text": doc["sections"].get(matched_section) if matched_section else None,
        "note": "all sellers operate under this ONE standard agreement template - no per-vendor bespoke contracts exist in this dataset",
    }


def check_policy_compliance(db: DatabricksClient, topic: str) -> dict:
    """Look up whether a topic is covered in the HR policy, and what it says."""
    doc = load_document("hr_policy.md")
    matched_section = find_section(doc, topic)

    return {
        "topic": topic,
        "covered": matched_section is not None,
        "matched_section": matched_section,
        "clause_text": doc["sections"].get(matched_section) if matched_section else None,
    }


def contract_expiry_tracker(db: DatabricksClient, renewal_window_days: int = 30) -> dict:
    """Sellers approaching their contract's annual renewal date, based on real onboarding dates."""
    sql = f"""
        SELECT oi.seller_id, MIN(o.order_purchase_timestamp) AS onboarding_date
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        GROUP BY oi.seller_id
    """
    rows = db.query(sql)

    cutoff_row = db.query(f"""
        SELECT MAX(order_purchase_timestamp) AS latest FROM {db.table('orders')}
    """)[0]
    cutoff_dt = datetime.fromisoformat(str(cutoff_row["latest"]).replace("Z", "").replace("T", " ")[:19])

    renewing_soon = []
    for r in rows:
        onboard_dt = datetime.fromisoformat(str(r["onboarding_date"]).replace("Z", "").replace("T", " ")[:19])
        years_elapsed = max(0, (cutoff_dt - onboard_dt).days // 365)
        next_anniversary = onboard_dt + timedelta(days=365 * (years_elapsed + 1))
        days_until_renewal = (next_anniversary - cutoff_dt).days
        if 0 <= days_until_renewal <= renewal_window_days:
            renewing_soon.append({
                "seller_id": r["seller_id"],
                "onboarding_date": str(r["onboarding_date"]),
                "next_renewal_date": next_anniversary.date().isoformat(),
                "days_until_renewal": days_until_renewal,
            })

    return {
        "renewing_soon": renewing_soon,
        "renewing_soon_count": len(renewing_soon),
        "total_sellers": len(rows),
        "renewal_window_days": renewal_window_days,
        "note": "renewal cadence (annual) read from the contract's Term and Renewal clause and applied as a fixed 365-day period from each seller's onboarding date",
    }


def policy_gap_analysis(db: DatabricksClient, topic: str = None) -> dict:
    """Which expected HR policy topics are documented vs. missing. Documentation gaps only - no employment decisions."""
    doc = load_document("hr_policy.md")
    actual_sections = set(doc["sections"].keys())

    if topic:
        covered = find_section(doc, topic) is not None
        return {"topic": topic, "covered": covered}

    covered_topics = [t for t in EXPECTED_HR_TOPICS if t in actual_sections]
    missing_topics = [t for t in EXPECTED_HR_TOPICS if t not in actual_sections]

    return {
        "covered_topics": covered_topics,
        "missing_topics": missing_topics,
        "coverage_pct": round(len(covered_topics) / len(EXPECTED_HR_TOPICS) * 100, 1),
        "note": "documentation-coverage check only - flags gaps for human review, does not recommend or make employment decisions",
    }


def cross_reference_sla_compliance(db: DatabricksClient, vendor: str) -> dict:
    """Compare a seller's actual on-time delivery rate against the contractual SLA threshold.

    The flagship cross-agent tool: pulls real Operations-domain delivery data and checks it
    against a contractual term parsed from the vendor contract text (not a hardcoded duplicate
    of the threshold, so the check always reflects what the contract actually says).
    """
    safe_vendor = _escape(vendor)
    seller_check = db.query(f"SELECT COUNT(*) AS n FROM {db.table('sellers')} WHERE seller_id = '{safe_vendor}'")
    if int(seller_check[0]["n"]) == 0:
        return {"vendor": vendor, "found": False, "note": "seller_id not found in sellers table"}

    sql = f"""
        SELECT
            COUNT(DISTINCT o.order_id) AS order_count,
            ROUND(AVG(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END) * 100, 2) AS on_time_pct
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE oi.seller_id = '{safe_vendor}'
          AND o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    """
    row = db.query(sql)[0]
    order_count = int(row["order_count"])
    on_time_pct = float(row["on_time_pct"]) if row["on_time_pct"] is not None else None

    contract = load_document("vendor_contract.md")
    sla_text = contract["sections"].get("Service Level Agreement (SLA)", "")
    threshold_match = re.search(r"no less than (\d+(?:\.\d+)?)%", sla_text)
    threshold_pct = float(threshold_match.group(1)) if threshold_match else None

    compliant = (on_time_pct >= threshold_pct) if (on_time_pct is not None and threshold_pct is not None) else None

    return {
        "vendor": vendor,
        "found": True,
        "actual_on_time_pct": on_time_pct,
        "contractual_threshold_pct": threshold_pct,
        "compliant": compliant,
        "delivered_order_count": order_count,
        "sla_clause_source": "vendor_contract.md - Service Level Agreement (SLA)",
        "note": "threshold parsed directly from the contract text, not hardcoded - stays in sync if the contract document changes",
    }
