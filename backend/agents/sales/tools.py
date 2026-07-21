"""Sales Agent tools — query orders/order_items via DatabricksClient.

Data reality notes (flagged explicitly per tool output, not hidden):

1. Olist has no web/browsing data (no cart, no page views) — there's no true top-of-funnel
   conversion to measure. conversion_funnel_analysis() proxies this with the order FULFILLMENT
   funnel (created -> ... -> delivered, vs. canceled/unavailable drop-off), which is a
   different thing and is flagged as such.
2. Olist's `customer_id` is scoped to a single ORDER, not a person — `customer_unique_id` is
   the actual person-level identifier. repeat_purchase_rate() uses customer_unique_id
   deliberately; using customer_id here would silently undercount repeat customers to zero.
3. seller_sales_ranking() is a pure revenue/volume view — deliberately overlaps with
   Operations' (future) seller_performance_score(), which covers reliability instead. Per
   CLAUDE.md this overlap is intentional, not a bug: volume vs. reliability are different
   lenses on the same seller.
4. Olist's order data collection trails off mid-2018 — the final months (Sep/Oct 2018) have
   a handful of orders each versus thousands in every prior month, a collection cutoff
   artifact, not a real demand collapse. query_revenue_by_period() filters out anything after
   backend.agents.common.get_analysis_cutoff() (shared with Growth and Risk) rather than
   reporting a false decline. An earlier version computed this cutoff from the caller's own
   LIMIT-windowed rows instead of the full order history - with a small enough `periods`
   argument, the cutoff months could dominate the window and drag the median down with them,
   letting a real collection-cutoff month slip through as "complete" data. Using the shared
   full-history-based helper fixes this at the source.
"""
from backend.agents.common import get_analysis_cutoff
from backend.db import DatabricksClient

NON_REVENUE_STATUSES = ("'canceled'", "'unavailable'")
_EXCLUDE_CLAUSE = f"o.order_status NOT IN ({', '.join(NON_REVENUE_STATUSES)})"

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def query_revenue_by_period(db: DatabricksClient, period_unit: str = "month", periods: int = 12) -> dict:
    """Revenue and order count per period (day/week/month), most recent first then sorted ascending.

    Filters out anything after get_analysis_cutoff() BEFORE applying the `periods` LIMIT, so a
    small `periods` window can never end up dominated by (and therefore blind to) the trailing
    collection-cutoff months - see the module docstring for the bug this replaced.
    """
    unit = period_unit if period_unit in ("day", "week", "month") else "month"
    cutoff = get_analysis_cutoff(db)
    sql = f"""
        SELECT
            date_trunc('{unit}', o.order_purchase_timestamp) AS period,
            ROUND(SUM(oi.price), 2) AS revenue,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        WHERE {_EXCLUDE_CLAUSE}
          AND o.order_purchase_timestamp < timestamp('{cutoff}')
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {int(periods)}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["period"])

    trend_direction = "flat"
    if len(rows) >= 2:
        first, last = float(rows[0]["revenue"]), float(rows[-1]["revenue"])
        delta_pct = (last - first) / first * 100 if first else 0.0
        trend_direction = "growing" if delta_pct > 5 else "declining" if delta_pct < -5 else "flat"

    return {"period_unit": unit, "periods": rows, "trend_direction": trend_direction, "analysis_cutoff": cutoff}


def calculate_aov(db: DatabricksClient, months: int = 12) -> dict:
    """Average order value overall and by month."""
    sql = f"""
        SELECT
            date_trunc('month', o.order_purchase_timestamp) AS month,
            ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2) AS aov,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {int(months)}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["month"])

    overall_sql = f"""
        SELECT ROUND(SUM(oi.price) / COUNT(DISTINCT o.order_id), 2) AS overall_aov
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        WHERE {_EXCLUDE_CLAUSE}
    """
    overall = db.query(overall_sql)[0]["overall_aov"]

    return {"overall_aov": float(overall or 0.0), "monthly": rows}


def sales_by_category(db: DatabricksClient, top_n: int = 10) -> dict:
    """Revenue and items sold by product category (English name where available)."""
    sql = f"""
        SELECT
            COALESCE(t.product_category_name_english, p.product_category_name) AS category,
            ROUND(SUM(oi.price), 2) AS revenue,
            COUNT(*) AS items_sold
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        JOIN {db.table('products')} p ON oi.product_id = p.product_id
        LEFT JOIN {db.table('product_category_translation')} t ON p.product_category_name = t.product_category_name
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY category
        ORDER BY revenue DESC
    """
    rows = db.query(sql)
    return {"top_categories": rows[:top_n], "categories_analyzed": len(rows)}


def seller_sales_ranking(db: DatabricksClient, top_n: int = 10) -> dict:
    """Top sellers by revenue and order volume (volume lens, not reliability)."""
    sql = f"""
        SELECT
            oi.seller_id,
            ROUND(SUM(oi.price), 2) AS revenue,
            COUNT(DISTINCT oi.order_id) AS order_count
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY oi.seller_id
        ORDER BY revenue DESC
        LIMIT {int(top_n)}
    """
    rows = db.query(sql)
    return {
        "top_sellers": rows,
        "note": "revenue/volume ranking only - seller reliability is a separate (future) Operations metric",
    }


def conversion_funnel_analysis(db: DatabricksClient) -> dict:
    """Order-status fulfillment funnel - a proxy, since Olist has no browse/cart data for a true top-of-funnel view."""
    sql = f"""
        SELECT order_status, COUNT(*) AS n
        FROM {db.table('orders')}
        GROUP BY order_status
    """
    rows = db.query(sql)
    total = sum(int(r["n"]) for r in rows)
    breakdown = {r["order_status"]: int(r["n"]) for r in rows}
    drop_off = breakdown.get("canceled", 0) + breakdown.get("unavailable", 0)

    return {
        "status_breakdown": breakdown,
        "total_orders": total,
        "drop_off_count": drop_off,
        "drop_off_pct": round(drop_off / total * 100, 2) if total else 0.0,
        "note": "proxies order FULFILLMENT funnel (placed -> delivered vs. canceled/unavailable) - Olist has no browse/cart data for a true top-of-funnel conversion view",
    }


def repeat_purchase_rate(db: DatabricksClient) -> dict:
    """Share of distinct customers (by customer_unique_id) with more than one order."""
    sql = f"""
        SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS order_count
        FROM {db.table('customers')} c
        JOIN {db.table('orders')} o ON c.customer_id = o.customer_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY c.customer_unique_id
    """
    rows = db.query(sql)
    total_customers = len(rows)
    repeat_customers = sum(1 for r in rows if int(r["order_count"]) > 1)

    return {
        "total_customers": total_customers,
        "repeat_customers": repeat_customers,
        "repeat_rate_pct": round(repeat_customers / total_customers * 100, 2) if total_customers else 0.0,
        "note": "uses customer_unique_id (person-level) not customer_id (order-level) - Olist issues a new customer_id per order",
    }


def seasonal_sales_pattern(db: DatabricksClient) -> dict:
    """Revenue by calendar month, aggregated across all years, to surface seasonality."""
    sql = f"""
        SELECT
            month(o.order_purchase_timestamp) AS month_num,
            ROUND(SUM(oi.price), 2) AS revenue,
            COUNT(DISTINCT o.order_id) AS order_count
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY month_num
        ORDER BY month_num
    """
    rows = db.query(sql)
    for r in rows:
        r["month_name"] = _MONTH_NAMES[int(r["month_num"]) - 1]

    peak = max(rows, key=lambda r: float(r["revenue"])) if rows else None
    trough = min(rows, key=lambda r: float(r["revenue"])) if rows else None

    return {
        "by_month": rows,
        "peak_month": peak["month_name"] if peak else None,
        "trough_month": trough["month_name"] if trough else None,
        "note": "aggregated across all years in the dataset (2016-2018) - not enough years to separate seasonality from year-over-year growth",
    }


def cross_sell_opportunities(db: DatabricksClient, min_co_occurrence: int = 20, top_n: int = 10) -> dict:
    """Category pairs frequently bought in the same order (co-occurrence, a simple market-basket view)."""
    category_expr = f"""
        SELECT DISTINCT oi.order_id,
               COALESCE(t.product_category_name_english, p.product_category_name) AS category
        FROM {db.table('order_items')} oi
        JOIN {db.table('products')} p ON oi.product_id = p.product_id
        LEFT JOIN {db.table('product_category_translation')} t ON p.product_category_name = t.product_category_name
    """
    sql = f"""
        SELECT c1.category AS category_a, c2.category AS category_b,
               COUNT(DISTINCT c1.order_id) AS orders_together
        FROM ({category_expr}) c1
        JOIN ({category_expr}) c2 ON c1.order_id = c2.order_id AND c1.category < c2.category
        GROUP BY category_a, category_b
        HAVING COUNT(DISTINCT c1.order_id) >= {int(min_co_occurrence)}
        ORDER BY orders_together DESC
        LIMIT {int(top_n)}
    """
    rows = db.query(sql)
    return {
        "category_pairs": rows,
        "min_co_occurrence_threshold": min_co_occurrence,
        "note": "co-occurrence within the same order, a simple market-basket proxy - not a trained recommendation model",
    }
