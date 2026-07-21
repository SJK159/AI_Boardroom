"""Growth Agent tools — query orders/customers/sellers via DatabricksClient.

Data reality notes (flagged explicitly per tool output, not hidden):

1. Olist's order data collection trails off mid-2018 (see Sales agent's finding of the same
   issue: Sept 2018 has 16 orders, Oct has 4, vs. thousands per month before that). Every
   trend/growth tool here calls backend.agents.common.get_analysis_cutoff() (shared with Sales
   and Risk) and excludes anything after it, rather than letting a collection artifact read as
   market collapse.
2. product_gap_analysis() is explicitly the weakest-fit tool for this dataset, per CLAUDE.md:
   Olist only records products actually SOLD on the platform - there is no external market
   data, no competitor catalog, no visibility into demand for products never listed at all.
   What's implemented here is a supply/demand imbalance proxy WITHIN existing categories
   (revenue-per-seller as a signal of under- vs over-supplied categories), not true
   whitespace/product-gap analysis. Flagged accordingly rather than overclaiming.
"""
from backend.agents.common import get_analysis_cutoff
from backend.db import DatabricksClient

NON_REVENUE_STATUSES = ("'canceled'", "'unavailable'")
_EXCLUDE_CLAUSE = f"o.order_status NOT IN ({', '.join(NON_REVENUE_STATUSES)})"


def regional_sales_breakdown(db: DatabricksClient, top_n: int = 10) -> dict:
    """Revenue and order count by customer state."""
    sql = f"""
        SELECT
            c.customer_state AS state,
            ROUND(SUM(oi.price), 2) AS revenue,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        JOIN {db.table('customers')} c ON o.customer_id = c.customer_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY c.customer_state
        ORDER BY revenue DESC
    """
    rows = db.query(sql)
    return {"by_state": rows, "top_states": rows[:top_n], "states_analyzed": len(rows)}


def _period_revenue_by_dimension(db: DatabricksClient, dimension_sql: str, dimension_alias: str, joins: str, cutoff: str) -> tuple[dict, dict]:
    """Shared recent-vs-prior-6-months revenue split, grouped by an arbitrary dimension."""
    recent_sql = f"""
        SELECT {dimension_sql} AS {dimension_alias}, ROUND(SUM(oi.price), 2) AS revenue
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        {joins}
        WHERE {_EXCLUDE_CLAUSE}
          AND o.order_purchase_timestamp < timestamp('{cutoff}')
          AND o.order_purchase_timestamp >= add_months(timestamp('{cutoff}'), -6)
        GROUP BY {dimension_alias}
    """
    prior_sql = f"""
        SELECT {dimension_sql} AS {dimension_alias}, ROUND(SUM(oi.price), 2) AS revenue
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        {joins}
        WHERE {_EXCLUDE_CLAUSE}
          AND o.order_purchase_timestamp < add_months(timestamp('{cutoff}'), -6)
          AND o.order_purchase_timestamp >= add_months(timestamp('{cutoff}'), -12)
        GROUP BY {dimension_alias}
    """
    recent = {r[dimension_alias]: float(r["revenue"]) for r in db.query(recent_sql)}
    prior = {r[dimension_alias]: float(r["revenue"]) for r in db.query(prior_sql)}
    return recent, prior


def market_expansion_signals(db: DatabricksClient, top_n: int = 5, min_prior_revenue: float = 5000) -> dict:
    """States with the strongest revenue growth, recent 6 months vs. the prior 6 - expansion signals.

    Filters out states below min_prior_revenue before ranking: a tiny baseline (e.g. R$650)
    can produce an 800%+ "growth" number that's really just noise, not a real signal.
    """
    cutoff = get_analysis_cutoff(db)
    joins = f"JOIN {db.table('customers')} c ON o.customer_id = c.customer_id"
    recent, prior = _period_revenue_by_dimension(db, "c.customer_state", "state", joins, cutoff)

    signals = []
    noise_filtered = 0
    for state, recent_rev in recent.items():
        prior_rev = prior.get(state, 0.0)
        if prior_rev > 0 and prior_rev < min_prior_revenue:
            noise_filtered += 1
            continue
        growth_pct = round((recent_rev - prior_rev) / prior_rev * 100, 2) if prior_rev > 0 else None
        signals.append({"state": state, "recent_revenue": recent_rev, "prior_revenue": prior_rev, "growth_pct": growth_pct})

    signals.sort(key=lambda s: (s["growth_pct"] is None, -(s["growth_pct"] or 0)))
    return {
        "signals": signals[:top_n],
        "analysis_cutoff": cutoff,
        "window": "recent 6 months vs. prior 6 months",
        "min_prior_revenue_threshold": min_prior_revenue,
        "noise_filtered_count": noise_filtered,
    }


def category_growth_rate(db: DatabricksClient, top_n: int = 10, min_prior_revenue: float = 2000) -> dict:
    """Category revenue growth, recent 6 months vs. the prior 6.

    Filters out categories below min_prior_revenue before ranking - same small-base-noise
    guard as market_expansion_signals (e.g. R$80 -> R$3,953 reads as 4849% "growth" but is
    really a near-empty category, not a real trend).
    """
    cutoff = get_analysis_cutoff(db)
    joins = f"""
        JOIN {db.table('products')} p ON oi.product_id = p.product_id
        LEFT JOIN {db.table('product_category_translation')} t ON p.product_category_name = t.product_category_name
    """
    recent, prior = _period_revenue_by_dimension(
        db, "COALESCE(t.product_category_name_english, p.product_category_name)", "category", joins, cutoff
    )

    growth = []
    noise_filtered = 0
    for category, recent_rev in recent.items():
        prior_rev = prior.get(category, 0.0)
        if prior_rev > 0 and prior_rev < min_prior_revenue:
            noise_filtered += 1
            continue
        growth_pct = round((recent_rev - prior_rev) / prior_rev * 100, 2) if prior_rev > 0 else None
        growth.append({"category": category, "recent_revenue": recent_rev, "prior_revenue": prior_rev, "growth_pct": growth_pct})

    growth.sort(key=lambda s: (s["growth_pct"] is None, -(s["growth_pct"] or 0)))
    return {
        "fastest_growing": growth[:top_n],
        "analysis_cutoff": cutoff,
        "window": "recent 6 months vs. prior 6 months",
        "min_prior_revenue_threshold": min_prior_revenue,
        "noise_filtered_count": noise_filtered,
    }


def customer_acquisition_trend(db: DatabricksClient, months: int = 12) -> dict:
    """New customers (by first order date, person-level via customer_unique_id) per month."""
    cutoff = get_analysis_cutoff(db)
    sql = f"""
        SELECT date_trunc('month', first_order) AS month, COUNT(*) AS new_customers
        FROM (
            SELECT c.customer_unique_id, MIN(o.order_purchase_timestamp) AS first_order
            FROM {db.table('orders')} o
            JOIN {db.table('customers')} c ON o.customer_id = c.customer_id
            WHERE {_EXCLUDE_CLAUSE}
            GROUP BY c.customer_unique_id
        ) t
        WHERE first_order < timestamp('{cutoff}')
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {int(months)}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["month"])

    trend_direction = "flat"
    if len(rows) >= 2:
        first, last = int(rows[0]["new_customers"]), int(rows[-1]["new_customers"])
        delta_pct = (last - first) / first * 100 if first else 0.0
        trend_direction = "growing" if delta_pct > 5 else "declining" if delta_pct < -5 else "flat"

    return {"monthly": rows, "trend_direction": trend_direction, "analysis_cutoff": cutoff}


def underperforming_region_diagnosis(db: DatabricksClient, bottom_n: int = 5, min_prior_revenue: float = 5000) -> dict:
    """States with negative revenue growth, ranked most-declining first.

    Same small-base-noise guard as market_expansion_signals: a state dropping from R$600 to
    R$50 reads as -92% but isn't a meaningful business signal at that volume.
    """
    cutoff = get_analysis_cutoff(db)
    joins = f"JOIN {db.table('customers')} c ON o.customer_id = c.customer_id"
    recent, prior = _period_revenue_by_dimension(db, "c.customer_state", "state", joins, cutoff)

    diagnosed = []
    for state, prior_rev in prior.items():
        recent_rev = recent.get(state, 0.0)
        if prior_rev < min_prior_revenue:
            continue
        growth_pct = round((recent_rev - prior_rev) / prior_rev * 100, 2)
        if growth_pct < 0:
            diagnosed.append({"state": state, "recent_revenue": recent_rev, "prior_revenue": prior_rev, "growth_pct": growth_pct})

    diagnosed.sort(key=lambda s: s["growth_pct"])
    return {
        "underperforming_states": diagnosed[:bottom_n],
        "states_declining": len(diagnosed),
        "analysis_cutoff": cutoff,
        "min_prior_revenue_threshold": min_prior_revenue,
        "note": "declining revenue (recent 6mo vs prior 6mo) among states with an established, non-trivial prior-period baseline",
    }


def new_seller_onboarding_rate(db: DatabricksClient, months: int = 12) -> dict:
    """New sellers (by first order date) per month."""
    cutoff = get_analysis_cutoff(db)
    sql = f"""
        SELECT date_trunc('month', first_order) AS month, COUNT(*) AS new_sellers
        FROM (
            SELECT oi.seller_id, MIN(o.order_purchase_timestamp) AS first_order
            FROM {db.table('order_items')} oi
            JOIN {db.table('orders')} o ON oi.order_id = o.order_id
            WHERE {_EXCLUDE_CLAUSE}
            GROUP BY oi.seller_id
        ) t
        WHERE first_order < timestamp('{cutoff}')
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {int(months)}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["month"])

    trend_direction = "flat"
    if len(rows) >= 2:
        first, last = int(rows[0]["new_sellers"]), int(rows[-1]["new_sellers"])
        delta_pct = (last - first) / first * 100 if first else 0.0
        trend_direction = "growing" if delta_pct > 5 else "declining" if delta_pct < -5 else "flat"

    return {"monthly": rows, "trend_direction": trend_direction, "analysis_cutoff": cutoff}


def product_gap_analysis(db: DatabricksClient, min_sellers: int = 1, top_n: int = 10) -> dict:
    """Supply/demand imbalance proxy: revenue-per-seller by category.

    NOT true product-gap/whitespace analysis - see module docstring. High revenue-per-seller
    suggests an undersupplied category (opportunity); low suggests oversaturation.
    """
    sql = f"""
        SELECT
            COALESCE(t.product_category_name_english, p.product_category_name) AS category,
            COUNT(DISTINCT oi.seller_id) AS seller_count,
            ROUND(SUM(oi.price), 2) AS revenue,
            ROUND(SUM(oi.price) / COUNT(DISTINCT oi.seller_id), 2) AS revenue_per_seller
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        JOIN {db.table('products')} p ON oi.product_id = p.product_id
        LEFT JOIN {db.table('product_category_translation')} t ON p.product_category_name = t.product_category_name
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY category
        HAVING COUNT(DISTINCT oi.seller_id) >= {int(min_sellers)}
        ORDER BY revenue_per_seller DESC
    """
    rows = db.query(sql)
    return {
        "potentially_undersupplied": rows[:top_n],
        "potentially_oversaturated": rows[-top_n:][::-1] if len(rows) > top_n else [],
        "categories_analyzed": len(rows),
        "note": "internal marketplace supply/demand proxy (revenue-per-seller) within EXISTING categories - not true product-gap/whitespace analysis, which would need external market data this dataset doesn't have. Weakest-fit tool in the Growth roster, per CLAUDE.md.",
    }
