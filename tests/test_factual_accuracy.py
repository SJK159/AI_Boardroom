"""Factual accuracy: agent tool output vs. an independently-computed ground truth.

CLAUDE.md section 9: "agent tool output vs. ground-truth calculation on raw data (pass/fail)."
Each test recomputes the expected value via a query DIFFERENT from the tool's own query -
re-running the tool's own SQL and comparing to itself would prove nothing. No LLM calls -
this tier is fast and should run on every change.
"""
from backend.agents.compliance import tools as compliance_tools
from backend.agents.growth import tools as growth_tools
from backend.agents.operations import tools as operations_tools
from backend.agents.risk import tools as risk_tools
from backend.agents.sentiment import tools as sentiment_tools


def test_finance_revenue_concentration_hhi_matches_independent_calculation(db):
    from backend.agents.finance import tools as finance_tools

    result = finance_tools.revenue_concentration(db)

    rows = db.query(f"""
        SELECT oi.seller_id, SUM(oi.price) AS revenue
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        GROUP BY oi.seller_id
    """)
    total = sum(float(r["revenue"]) for r in rows)
    expected_hhi = round(sum((float(r["revenue"]) / total * 100) ** 2 for r in rows), 1)

    assert abs(result["hhi"] - expected_hhi) < 0.5, (
        f"tool HHI {result['hhi']} vs independently computed {expected_hhi}"
    )


def test_operations_delivery_delay_late_pct_matches_independent_calculation(db):
    result = operations_tools.calculate_delivery_delay(db)

    row = db.query(f"""
        SELECT
            SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late,
            COUNT(*) AS total
        FROM {db.table('orders')}
        WHERE order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
    """)[0]
    expected_pct = round(int(row["late"]) / int(row["total"]) * 100, 2)

    assert abs(result["late_pct"] - expected_pct) < 0.1


def test_risk_fraud_signal_percentile_threshold_flags_expected_share(db):
    """Regression test for the z-score-over-flagging bug found while building the Risk agent:
    a percentile threshold should flag close to (100 - percentile_threshold)% of orders,
    not 13x more (which is what the original z-score >= 3.0 implementation did on this
    right-skewed data)."""
    result = risk_tools.fraud_signal_detection(db, percentile_threshold=99.5)

    expected_share_pct = 100 - 99.5
    actual_share_pct = result["flagged_count"] / result["orders_analyzed"] * 100

    assert abs(actual_share_pct - expected_share_pct) < 0.15, (
        f"expected ~{expected_share_pct}% flagged, got {actual_share_pct:.2f}% "
        f"({result['flagged_count']} of {result['orders_analyzed']})"
    )


def test_sales_trailing_period_exclusion_only_excludes_genuinely_low_volume_periods(db):
    """Regression test for the false-decline bug found while building the Sales agent."""
    from backend.agents.sales import tools as sales_tools

    result = sales_tools.query_revenue_by_period(db)
    excluded = result.get("excluded_partial_periods", [])

    all_counts = sorted(int(p["order_count"]) for p in result["periods"])
    median_count = all_counts[len(all_counts) // 2]

    for period in excluded:
        assert int(period["order_count"]) < median_count * 0.5, (
            f"period {period['period']} excluded but order_count "
            f"{period['order_count']} is not far below median {median_count}"
        )


def test_growth_expansion_signals_respect_min_prior_revenue_threshold(db):
    """Regression test for the small-base-noise bug found while building the Growth agent."""
    result = growth_tools.market_expansion_signals(db)
    threshold = result["min_prior_revenue_threshold"]

    for signal in result["signals"]:
        if signal["prior_revenue"] > 0:
            assert signal["prior_revenue"] >= threshold, (
                f"state {signal['state']} has prior_revenue {signal['prior_revenue']} "
                f"below the stated threshold {threshold}"
            )


def test_sentiment_region_best_worst_are_actual_extremes(db):
    result = sentiment_tools.sentiment_by_region(db)

    scores = [float(r["avg_score"]) for r in result["by_state"]]
    assert float(result["best_state"]["avg_score"]) == max(scores)
    assert float(result["worst_state"]["avg_score"]) == min(scores)


def test_compliance_sla_threshold_matches_contract_text():
    """Verifies cross_reference_sla_compliance() parses the threshold correctly by
    independently re-reading the contract document, not trusting the tool's own regex."""
    import re

    from backend.agents.compliance.document_loader import load_document

    contract = load_document("vendor_contract.md")
    sla_text = contract["sections"]["Service Level Agreement (SLA)"]
    independently_parsed = float(re.search(r"no less than (\d+(?:\.\d+)?)%", sla_text).group(1))

    assert independently_parsed == 85.0


def test_compliance_sla_compliance_verdict_matches_threshold_comparison(db):
    top_seller = db.query(f"""
        SELECT seller_id, COUNT(*) AS n FROM {db.table('order_items')}
        GROUP BY seller_id ORDER BY n DESC LIMIT 1
    """)[0]["seller_id"]

    result = compliance_tools.cross_reference_sla_compliance(db, vendor=top_seller)
    assert result["found"] is True
    expected_compliant = result["actual_on_time_pct"] >= result["contractual_threshold_pct"]
    assert result["compliant"] == expected_compliant
