"""Finance Agent tools — each queries Delta tables via DatabricksClient and
returns raw computed data. FinanceAgent.analyze() wraps each call in
_call_tool() and turns the result into a Finding.

Data limitation note: the Olist dataset has no cost/COGS field (products table
has only physical attributes, not unit cost) and no explicit payment-failure or
refund flag. Tools that would need those fields use the closest available proxy
and say so explicitly in their output — see calculate_cogs(), payment_failure_rate(),
and refund_impact_analysis().
"""
import numpy as np

from backend.db import DatabricksClient

NON_REVENUE_STATUSES = ("'canceled'", "'unavailable'")
_EXCLUDE_CLAUSE = f"order_status NOT IN ({', '.join(NON_REVENUE_STATUSES)})"


def calculate_margin_trend(db: DatabricksClient, months: int = 12) -> dict:
    """Monthly contribution margin (price - freight_value) as a trend.

    True gross margin needs COGS, which Olist doesn't provide — this uses
    freight cost as the only real cost signal available per order line.
    """
    sql = f"""
        SELECT
            date_trunc('month', o.order_purchase_timestamp) AS month,
            ROUND(SUM(oi.price), 2) AS revenue,
            ROUND(SUM(oi.freight_value), 2) AS freight_cost,
            ROUND(SUM(oi.price) - SUM(oi.freight_value), 2) AS contribution_margin
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {months}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["month"])
    for r in rows:
        r["margin_pct"] = round(float(r["contribution_margin"]) / float(r["revenue"]) * 100, 2) if float(r["revenue"]) else 0.0

    trend_direction = "flat"
    if len(rows) >= 2:
        delta = rows[-1]["margin_pct"] - rows[0]["margin_pct"]
        trend_direction = "improving" if delta > 1 else "declining" if delta < -1 else "flat"

    return {"monthly": rows, "trend_direction": trend_direction, "note": "proxy: contribution margin = price - freight, no COGS data in source"}


def detect_revenue_anomalies(db: DatabricksClient, std_threshold: float = 2.0) -> dict:
    """Flag days where revenue is more than `std_threshold` std devs from the mean."""
    sql = f"""
        SELECT
            date_trunc('day', o.order_purchase_timestamp) AS day,
            ROUND(SUM(oi.price), 2) AS revenue
        FROM {db.table('orders')} o
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY 1
        ORDER BY 1
    """
    rows = db.query(sql)
    values = np.array([float(r["revenue"]) for r in rows])
    mean, std = float(values.mean()), float(values.std())

    anomalies = []
    for r, v in zip(rows, values):
        z = (v - mean) / std if std else 0.0
        if abs(z) >= std_threshold:
            anomalies.append({"day": str(r["day"]), "revenue": v, "z_score": round(z, 2)})

    return {"mean_daily_revenue": round(mean, 2), "std_daily_revenue": round(std, 2), "anomalies": anomalies, "days_analyzed": len(rows)}


def payment_failure_rate(db: DatabricksClient) -> dict:
    """Share of orders that never completed, using order_status as a proxy.

    Olist has no explicit 'payment failed' flag — 'canceled' and 'unavailable'
    are the closest signal, but can also reflect stock or logistics issues,
    not payment failure specifically.
    """
    sql = f"""
        SELECT order_status, COUNT(*) AS n
        FROM {db.table('orders')}
        GROUP BY order_status
    """
    rows = db.query(sql)
    total = sum(int(r["n"]) for r in rows)
    failed = sum(int(r["n"]) for r in rows if r["order_status"] in ("canceled", "unavailable"))
    rate = round(failed / total * 100, 2) if total else 0.0

    return {
        "status_breakdown": {r["order_status"]: int(r["n"]) for r in rows},
        "failure_rate_pct": rate,
        "total_orders": total,
        "note": "proxy: canceled + unavailable orders, no explicit payment-failure field in source",
    }


def calculate_cogs(db: DatabricksClient) -> dict:
    """COGS is not computable from this dataset — flagged, not faked."""
    return {
        "available": False,
        "reason": "Olist products table has physical attributes only (weight, dimensions, photo count) — no unit cost or COGS field exists in the source data.",
        "closest_proxy": "freight_value is the only real per-order cost signal; used in calculate_margin_trend() as a contribution-margin proxy.",
    }


def cash_flow_forecast(db: DatabricksClient, periods_ahead: int = 3) -> dict:
    """Linear-trend forecast of net cash-in (completed order payments) for the next N months."""
    sql = f"""
        SELECT
            date_trunc('month', o.order_purchase_timestamp) AS month,
            ROUND(SUM(p.payment_value), 2) AS net_cash_in
        FROM {db.table('orders')} o
        JOIN {db.table('order_payments')} p ON o.order_id = p.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY 1
        ORDER BY 1
    """
    rows = db.query(sql)
    y = np.array([float(r["net_cash_in"]) for r in rows])
    x = np.arange(len(y))

    if len(y) < 2:
        return {"history": rows, "forecast": [], "note": "insufficient history for a trend line"}

    slope, intercept = np.polyfit(x, y, 1)
    forecast = [
        {"period_offset": i, "projected_cash_in": round(float(slope * (len(y) + i - 1) + intercept), 2)}
        for i in range(1, periods_ahead + 1)
    ]

    return {
        "history": rows,
        "forecast": forecast,
        "monthly_trend_slope": round(float(slope), 2),
        "note": "linear projection from historical monthly totals — a simple trend estimate, not a full cash-flow model (no AR/AP timing data available)",
    }


def refund_impact_analysis(db: DatabricksClient) -> dict:
    """Financial exposure from canceled/unavailable orders, as a refund proxy.

    Olist has no explicit refund field — this treats payment value on
    canceled/unavailable orders as the closest available signal.
    """
    sql = f"""
        SELECT
            SUM(CASE WHEN o.order_status IN ('canceled', 'unavailable') THEN p.payment_value ELSE 0 END) AS at_risk_value,
            SUM(p.payment_value) AS total_value,
            COUNT(DISTINCT CASE WHEN o.order_status IN ('canceled', 'unavailable') THEN o.order_id END) AS at_risk_orders
        FROM {db.table('orders')} o
        JOIN {db.table('order_payments')} p ON o.order_id = p.order_id
    """
    row = db.query(sql)[0]
    total = float(row["total_value"]) or 1.0
    at_risk = float(row["at_risk_value"]) or 0.0

    return {
        "at_risk_value": round(at_risk, 2),
        "total_value": round(total, 2),
        "impact_pct": round(at_risk / total * 100, 2),
        "at_risk_orders": int(row["at_risk_orders"] or 0),
        "note": "proxy: payment value on canceled/unavailable orders, no explicit refund field in source",
    }


def revenue_concentration(db: DatabricksClient, top_n: int = 10) -> dict:
    """Seller revenue concentration via top-N share and Herfindahl-Hirschman Index."""
    sql = f"""
        SELECT oi.seller_id, ROUND(SUM(oi.price), 2) AS revenue
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE {_EXCLUDE_CLAUSE}
        GROUP BY oi.seller_id
        ORDER BY revenue DESC
    """
    rows = db.query(sql)
    total = sum(float(r["revenue"]) for r in rows) or 1.0
    top_n_revenue = sum(float(r["revenue"]) for r in rows[:top_n])
    hhi = sum((float(r["revenue"]) / total * 100) ** 2 for r in rows)

    concentration_level = "high" if hhi > 1500 else "moderate" if hhi > 1000 else "low"

    return {
        "total_sellers": len(rows),
        "top_n": top_n,
        "top_n_revenue_share_pct": round(top_n_revenue / total * 100, 2),
        "hhi": round(hhi, 1),
        "concentration_level": concentration_level,
        "top_sellers": rows[:top_n],
    }
