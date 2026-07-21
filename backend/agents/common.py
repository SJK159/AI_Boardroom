"""Shared helpers used by more than one specialist agent's tools.py.

Deliberately a narrow exception to the "each agent's tools.py is self-contained" convention:
this specific heuristic was independently duplicated 3 times (Sales, Growth, Risk) and one of
those copies drifted into a real bug (Sales computed the median from an already-windowed row
set instead of the full history, letting the trailing-cutoff months it was meant to detect
drag the threshold down with them). Centralizing it here means a future fix only has to happen
once, and the median is always computed from the full order history regardless of what window
a caller later asks for.
"""
from backend.db import DatabricksClient


def get_analysis_cutoff(db: DatabricksClient) -> str:
    """Timestamp literal marking the end of the last COMPLETE month in the data.

    Detects trailing months whose order count is far below the FULL DATASET's median (a
    collection cutoff, not a real drop-off) and excludes them. Always computed from the
    complete order history, never from a caller-limited window - a window that happens to be
    dominated by the anomalous trailing months would otherwise drag the median down with them
    and fail to detect the cutoff (the bug this module was extracted to fix).
    """
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
