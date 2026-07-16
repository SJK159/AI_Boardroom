"""Risk Agent tools — query orders/order_payments via DatabricksClient.

Data reality notes (flagged explicitly per tool output, not hidden):

1. Olist has no chargeback/dispute field. flag_payment_disputes() proxies this with canceled
   orders that had payment already collected - money taken, order not fulfilled is the
   closest available signal, not a real dispute record.
2. Olist has no fraud label. fraud_signal_detection() flags statistical outliers in per-order
   payment value (z-score) as SIGNALS worth reviewing, not confirmed fraud.
3. customer_churn_prediction() is a heuristic (time-since-last-order vs. a customer's own
   historical order gap), not a trained predictive model. "Now" is the dataset's own analysis
   cutoff (mid-2018), not the real current date - this is historical data, and computing
   recency against today's date would flag every customer as churned.
4. regulatory_exposure_check() is explicitly NOT AVAILABLE from this dataset - Olist's
   transactional data has no legal/compliance signals at all (no consent records, no KYC/AML
   fields, no tax jurisdiction data). Real regulatory exposure checking depends on the
   Compliance/HR agent's institutional documents (company registration, vendor contracts),
   which don't exist yet.
5. concentration_risk() is deliberately CUSTOMER-side concentration (revenue dependency on
   top customers) - Finance's revenue_concentration() already covers seller-side
   concentration. Different lens, same HHI methodology.
6. customer_churn_prediction() (forward-looking, per-customer risk) is a different lens from
   Sales' repeat_purchase_rate() (historical, aggregate rate) - related but not duplicative.
"""
from backend.db import DatabricksClient

import numpy as np


def _get_analysis_cutoff(db: DatabricksClient) -> str:
    """Same trailing-partial-period detection used by Sales/Growth - see their tools.py."""
    sql = f"""
        SELECT date_trunc('month', order_purchase_timestamp) AS month, COUNT(*) AS n
        FROM {db.table('orders')}
        GROUP BY 1 ORDER BY 1
    """
    rows = db.query(sql)
    counts = sorted(int(r["n"]) for r in rows)
    median_count = counts[len(counts) // 2] if counts else 0
    complete = list(rows)
    while complete and median_count and int(complete[-1]["n"]) < median_count * 0.5:
        complete.pop()
    if not complete:
        return "9999-01-01"
    return str(complete[-1]["month"])


def cancellation_trend(db: DatabricksClient, months: int = 12) -> dict:
    """Monthly cancellation rate and trend direction."""
    cutoff = _get_analysis_cutoff(db)
    sql = f"""
        SELECT
            date_trunc('month', order_purchase_timestamp) AS month,
            COUNT(*) AS total_orders,
            SUM(CASE WHEN order_status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders
        FROM {db.table('orders')}
        WHERE order_purchase_timestamp < timestamp('{cutoff}')
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {int(months)}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["month"])
    for r in rows:
        r["cancellation_pct"] = round(int(r["canceled_orders"]) / int(r["total_orders"]) * 100, 2) if int(r["total_orders"]) else 0.0

    trend_direction = "flat"
    if len(rows) >= 2:
        delta = rows[-1]["cancellation_pct"] - rows[0]["cancellation_pct"]
        trend_direction = "worsening" if delta > 1 else "improving" if delta < -1 else "flat"

    return {"monthly": rows, "trend_direction": trend_direction, "analysis_cutoff": cutoff}


def flag_payment_disputes(db: DatabricksClient) -> dict:
    """Canceled orders with payment already collected - proxy for dispute/chargeback risk."""
    sql = f"""
        SELECT
            COUNT(DISTINCT o.order_id) AS at_risk_orders,
            ROUND(SUM(p.payment_value), 2) AS at_risk_value
        FROM {db.table('orders')} o
        JOIN {db.table('order_payments')} p ON o.order_id = p.order_id
        WHERE o.order_status = 'canceled'
    """
    row = db.query(sql)[0]
    return {
        "at_risk_orders": int(row["at_risk_orders"] or 0),
        "at_risk_value": float(row["at_risk_value"] or 0),
        "note": "proxy: canceled orders with payment already collected - Olist has no explicit dispute/chargeback field",
    }


def seller_churn_risk(db: DatabricksClient, inactive_months_threshold: int = 6) -> dict:
    """Sellers with no orders in the last N months relative to the analysis cutoff."""
    cutoff = _get_analysis_cutoff(db)
    sql = f"""
        SELECT oi.seller_id, MAX(o.order_purchase_timestamp) AS last_order_date, COUNT(DISTINCT o.order_id) AS total_orders
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        GROUP BY oi.seller_id
    """
    rows = db.query(sql)
    total_sellers = len(rows)
    cutoff_ts_sql = f"add_months(timestamp('{cutoff}'), -{int(inactive_months_threshold)})"
    threshold_row = db.query(f"SELECT {cutoff_ts_sql} AS threshold_date")[0]
    threshold_date = str(threshold_row["threshold_date"])

    at_risk = [r for r in rows if str(r["last_order_date"]) < threshold_date]
    at_risk.sort(key=lambda r: r["last_order_date"])

    return {
        "at_risk_sellers": at_risk[:20],
        "at_risk_count": len(at_risk),
        "total_sellers": total_sellers,
        "at_risk_pct": round(len(at_risk) / total_sellers * 100, 2) if total_sellers else 0.0,
        "inactive_months_threshold": inactive_months_threshold,
        "analysis_cutoff": cutoff,
    }


def concentration_risk(db: DatabricksClient, top_n: int = 10) -> dict:
    """Customer-side revenue concentration (HHI) - a different lens from Finance's seller-side revenue_concentration()."""
    sql = f"""
        SELECT c.customer_unique_id, ROUND(SUM(oi.price), 2) AS revenue
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        JOIN {db.table('customers')} c ON o.customer_id = c.customer_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        GROUP BY c.customer_unique_id
        ORDER BY revenue DESC
    """
    rows = db.query(sql)
    total = sum(float(r["revenue"]) for r in rows) or 1.0
    top_n_revenue = sum(float(r["revenue"]) for r in rows[:top_n])
    hhi = sum((float(r["revenue"]) / total * 100) ** 2 for r in rows)
    concentration_level = "high" if hhi > 1500 else "moderate" if hhi > 1000 else "low"

    return {
        "total_customers": len(rows),
        "top_n": top_n,
        "top_n_revenue_share_pct": round(top_n_revenue / total * 100, 2),
        "hhi": round(hhi, 1),
        "concentration_level": concentration_level,
        "note": "customer-side concentration - see Finance's revenue_concentration() for the seller-side view",
    }


def fraud_signal_detection(db: DatabricksClient, percentile_threshold: float = 99.5) -> dict:
    """Top-percentile outliers in per-order payment value - screening signals, not confirmed fraud.

    Uses a percentile threshold, not z-score: per-order payment value here is heavily
    right-skewed (skewness ~9), so z-score systematically over-flags (mean+3std caught 1.7%
    of orders in testing, versus the ~0.13% a normal distribution would predict) - the
    skewness makes "z >= 3" not actually rare on this data. A percentile threshold makes no
    assumption about the distribution's shape.
    """
    sql = f"""
        SELECT o.order_id, ROUND(SUM(p.payment_value), 2) AS total_payment
        FROM {db.table('orders')} o
        JOIN {db.table('order_payments')} p ON o.order_id = p.order_id
        GROUP BY o.order_id
    """
    rows = db.query(sql)
    values = np.array([float(r["total_payment"]) for r in rows])
    threshold_value = float(np.percentile(values, percentile_threshold))

    outliers = [
        {"order_id": r["order_id"], "total_payment": v}
        for r, v in zip(rows, values) if v >= threshold_value
    ]
    outliers.sort(key=lambda o: -o["total_payment"])

    return {
        "mean_order_value": round(float(values.mean()), 2),
        "median_order_value": round(float(np.median(values)), 2),
        "threshold_value": round(threshold_value, 2),
        "percentile_threshold": percentile_threshold,
        "flagged_orders": outliers[:20],
        "flagged_count": len(outliers),
        "orders_analyzed": len(rows),
        "note": "top-percentile payment values only - a screening signal, not confirmed fraud. Olist has no fraud label to validate against.",
    }


def customer_churn_prediction(db: DatabricksClient, min_orders: int = 2, overdue_multiplier: float = 1.5) -> dict:
    """Repeat customers overdue for their next order, relative to their own historical order gap."""
    cutoff = _get_analysis_cutoff(db)
    sql = f"""
        SELECT c.customer_unique_id, o.order_purchase_timestamp
        FROM {db.table('orders')} o
        JOIN {db.table('customers')} c ON o.customer_id = c.customer_id
        WHERE o.order_status NOT IN ('canceled', 'unavailable')
        ORDER BY c.customer_unique_id, o.order_purchase_timestamp
    """
    rows = db.query(sql)

    from collections import defaultdict
    from datetime import datetime

    by_customer = defaultdict(list)
    for r in rows:
        by_customer[r["customer_unique_id"]].append(r["order_purchase_timestamp"])

    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "").replace("T", " ")[:19])
    at_risk = []
    repeat_customer_count = 0

    for customer_id, timestamps in by_customer.items():
        if len(timestamps) < min_orders:
            continue
        repeat_customer_count += 1
        dts = sorted(datetime.fromisoformat(str(t).replace("Z", "").replace("T", " ")[:19]) for t in timestamps)
        gaps = [(dts[i + 1] - dts[i]).days for i in range(len(dts) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        days_since_last = (cutoff_dt - dts[-1]).days

        if avg_gap > 0 and days_since_last > avg_gap * overdue_multiplier:
            at_risk.append({
                "customer_unique_id": customer_id,
                "order_count": len(timestamps),
                "avg_gap_days": round(avg_gap, 1),
                "days_since_last_order": days_since_last,
            })

    at_risk.sort(key=lambda c: -(c["days_since_last_order"] / max(c["avg_gap_days"], 1)))

    return {
        "at_risk_customers": at_risk[:20],
        "at_risk_count": len(at_risk),
        "repeat_customers_analyzed": repeat_customer_count,
        "at_risk_pct_of_repeat": round(len(at_risk) / repeat_customer_count * 100, 2) if repeat_customer_count else 0.0,
        "analysis_cutoff": cutoff,
        "note": "heuristic (time-since-last-order vs. own historical gap), not a trained predictive model. 'Now' is the dataset's own cutoff, not today's date.",
    }


def regulatory_exposure_check(db: DatabricksClient) -> dict:
    """Not computable - Olist's transactional data has no legal/compliance signals at all."""
    return {
        "available": False,
        "reason": "Olist's schema has no regulatory/legal fields - no consent records, no KYC/AML data, no tax jurisdiction information.",
        "closest_proxy": "Real regulatory exposure checking depends on the Compliance/HR agent's institutional documents (company registration, vendor contracts, HR policy) - not built yet.",
    }
