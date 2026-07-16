"""Operations/Logistics Agent tools — query orders/order_items via DatabricksClient.

Data reality notes (flagged explicitly per tool output, not hidden):

1. Olist has no carrier/shipping-company field anywhere in the dataset — orders are fulfilled
   through Olist's logistics network but which carrier handled a given shipment isn't
   recorded. carrier_performance_comparison() explicitly returns "not available" rather than
   fabricating a comparison, the same treatment Finance gives calculate_cogs().
2. seller_performance_score() is a pure reliability view (on-time %, delay) — deliberately
   overlaps with Sales' seller_sales_ranking() (revenue/volume), which is a different lens on
   the same sellers. Per CLAUDE.md this overlap is intentional.
3. shipping_cost_analysis() looks at freight from a logistics-efficiency angle (interstate vs.
   intrastate cost, freight as % of item price) — deliberately distinct from Finance's
   calculate_margin_trend(), which uses freight as a cost-of-margin proxy. Same underlying
   field, different question.
4. All delay/duration tools use datediff or unix-timestamp differences on Olist's own
   estimated_delivery_date field — an internal Olist prediction, not a customer-facing SLA
   commitment from a named carrier.
"""
from backend.db import DatabricksClient

_DELIVERED_CLAUSE = (
    "o.order_status = 'delivered' "
    "AND o.order_delivered_customer_date IS NOT NULL "
    "AND o.order_estimated_delivery_date IS NOT NULL"
)


def calculate_delivery_delay(db: DatabricksClient) -> dict:
    """Average delay (actual vs. estimated delivery date) and share of late deliveries."""
    sql = f"""
        SELECT
            ROUND(AVG(datediff(o.order_delivered_customer_date, o.order_estimated_delivery_date)), 2) AS avg_delay_days,
            SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_count,
            COUNT(*) AS total_delivered
        FROM {db.table('orders')} o
        WHERE {_DELIVERED_CLAUSE}
    """
    row = db.query(sql)[0]
    total = int(row["total_delivered"])
    return {
        "avg_delay_days": float(row["avg_delay_days"]) if row["avg_delay_days"] is not None else 0.0,
        "late_count": int(row["late_count"]),
        "total_delivered": total,
        "late_pct": round(int(row["late_count"]) / total * 100, 2) if total else 0.0,
    }


def seller_performance_score(db: DatabricksClient, min_orders: int = 5, top_n: int = 10) -> dict:
    """Seller reliability: on-time delivery rate and average delay. Not a revenue metric."""
    sql = f"""
        SELECT
            oi.seller_id,
            COUNT(DISTINCT o.order_id) AS order_count,
            ROUND(AVG(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END) * 100, 2) AS on_time_pct,
            ROUND(AVG(datediff(o.order_delivered_customer_date, o.order_estimated_delivery_date)), 2) AS avg_delay_days
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        WHERE {_DELIVERED_CLAUSE}
        GROUP BY oi.seller_id
        HAVING COUNT(DISTINCT o.order_id) >= {int(min_orders)}
        ORDER BY on_time_pct DESC
    """
    rows = db.query(sql)
    return {
        "most_reliable": rows[:top_n],
        "least_reliable": rows[-top_n:][::-1] if len(rows) > top_n else [],
        "sellers_analyzed": len(rows),
        "min_orders_threshold": min_orders,
        "note": "reliability lens (on-time %, delay) - not a revenue/volume metric, see Sales' seller_sales_ranking for that",
    }


def flag_late_shipments(db: DatabricksClient, severe_threshold_days: int = 7) -> dict:
    """Share of deliveries that are mildly vs. severely late."""
    sql = f"""
        SELECT
            SUM(CASE WHEN datediff(o.order_delivered_customer_date, o.order_estimated_delivery_date) BETWEEN 1 AND {int(severe_threshold_days)} THEN 1 ELSE 0 END) AS mild_late,
            SUM(CASE WHEN datediff(o.order_delivered_customer_date, o.order_estimated_delivery_date) > {int(severe_threshold_days)} THEN 1 ELSE 0 END) AS severe_late,
            COUNT(*) AS total_delivered
        FROM {db.table('orders')} o
        WHERE {_DELIVERED_CLAUSE}
    """
    row = db.query(sql)[0]
    total = int(row["total_delivered"])
    mild, severe = int(row["mild_late"]), int(row["severe_late"])
    return {
        "mild_late_count": mild,
        "severe_late_count": severe,
        "total_delivered": total,
        "mild_late_pct": round(mild / total * 100, 2) if total else 0.0,
        "severe_late_pct": round(severe / total * 100, 2) if total else 0.0,
        "severe_threshold_days": severe_threshold_days,
    }


def shipping_cost_analysis(db: DatabricksClient) -> dict:
    """Freight cost from a logistics-efficiency angle: interstate vs. intrastate, freight as % of price."""
    sql = f"""
        SELECT
            CASE WHEN s.seller_state = c.customer_state THEN 'intrastate' ELSE 'interstate' END AS shipment_type,
            ROUND(AVG(oi.freight_value), 2) AS avg_freight,
            ROUND(AVG(oi.freight_value / oi.price) * 100, 2) AS avg_freight_pct_of_price,
            COUNT(*) AS item_count
        FROM {db.table('order_items')} oi
        JOIN {db.table('orders')} o ON oi.order_id = o.order_id
        JOIN {db.table('sellers')} s ON oi.seller_id = s.seller_id
        JOIN {db.table('customers')} c ON o.customer_id = c.customer_id
        WHERE oi.price > 0
        GROUP BY shipment_type
    """
    rows = db.query(sql)
    return {
        "by_shipment_type": rows,
        "note": "logistics-efficiency lens (freight relative to interstate/intrastate) - distinct from Finance's margin-impact framing of the same freight_value field",
    }


def carrier_performance_comparison(db: DatabricksClient) -> dict:
    """Not computable - Olist records no carrier/shipping-company identifier at all."""
    return {
        "available": False,
        "reason": "Olist's schema has no carrier or shipping-company field anywhere - shipments are fulfilled through Olist's logistics network but which carrier handled a given order isn't recorded.",
        "closest_proxy": "order_delivered_carrier_date -> order_delivered_customer_date transit time is tracked in fulfillment_bottleneck_detection(), but without a carrier identifier there's nothing to compare across carriers.",
    }


def fulfillment_bottleneck_detection(db: DatabricksClient) -> dict:
    """Average time spent in each fulfillment stage, in fractional days, to identify the slowest step."""
    sql = f"""
        SELECT
            ROUND(AVG((unix_timestamp(o.order_approved_at) - unix_timestamp(o.order_purchase_timestamp)) / 86400.0), 2) AS avg_approval_days,
            ROUND(AVG((unix_timestamp(o.order_delivered_carrier_date) - unix_timestamp(o.order_approved_at)) / 86400.0), 2) AS avg_handoff_days,
            ROUND(AVG((unix_timestamp(o.order_delivered_customer_date) - unix_timestamp(o.order_delivered_carrier_date)) / 86400.0), 2) AS avg_transit_days
        FROM {db.table('orders')} o
        WHERE o.order_status = 'delivered'
          AND o.order_approved_at IS NOT NULL
          AND o.order_delivered_carrier_date IS NOT NULL
          AND o.order_delivered_customer_date IS NOT NULL
    """
    row = db.query(sql)[0]
    stages = {
        "approval (purchase -> approved)": float(row["avg_approval_days"] or 0),
        "handoff (approved -> carrier pickup)": float(row["avg_handoff_days"] or 0),
        "transit (carrier pickup -> customer)": float(row["avg_transit_days"] or 0),
    }
    bottleneck = max(stages, key=stages.get)
    return {"stage_durations_days": stages, "bottleneck_stage": bottleneck}


def estimated_vs_actual_delivery_accuracy(db: DatabricksClient) -> dict:
    """How accurate Olist's own estimated_delivery_date is: mean absolute error and bias."""
    sql = f"""
        SELECT datediff(o.order_delivered_customer_date, o.order_estimated_delivery_date) AS diff_days
        FROM {db.table('orders')} o
        WHERE {_DELIVERED_CLAUSE}
    """
    rows = db.query(sql)
    diffs = [int(r["diff_days"]) for r in rows]
    if not diffs:
        return {"reviews_analyzed": 0, "mae_days": 0.0, "bias_days": 0.0, "within_3_days_pct": 0.0}

    mae = sum(abs(d) for d in diffs) / len(diffs)
    bias = sum(diffs) / len(diffs)
    within_3 = sum(1 for d in diffs if abs(d) <= 3) / len(diffs) * 100

    return {
        "orders_analyzed": len(diffs),
        "mae_days": round(mae, 2),
        "bias_days": round(bias, 2),
        "within_3_days_pct": round(within_3, 2),
        "note": "bias > 0 means deliveries tend to arrive LATER than Olist's own estimate (worse); bias < 0 means Olist's estimate is conservative and deliveries tend to arrive EARLY (under-promise, over-deliver)",
    }
