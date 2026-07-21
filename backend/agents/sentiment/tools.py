"""Sentiment Agent tools — query review data via DatabricksClient.

Data reality notes (flagged explicitly per tool output, not hidden):

1. search_reviews() does real semantic search via the local RAG layer (backend/rag/,
   multilingual embeddings — see build order step 6). It queries in whatever language the
   user writes in and still matches Portuguese review text, since the embedding model shares
   one vector space across languages; no translation step needed. Requires the review index
   to be built first — see notebooks/10_rag_index_build.ipynb.
2. extract_common_complaints() does raw word-frequency on Portuguese text with basic
   stopword filtering, not real NLP entity/topic extraction (that's the NLP/DL layer,
   step 2, not built yet).
3. review_answer_timestamp in Olist is when the CUSTOMER answered the satisfaction
   survey Olist sent them — not a seller's response time to a complaint. There is no
   seller-response-time field in this dataset. review_response_time_correlation()
   measures survey-answer latency, and is named/flagged accordingly.
4. Olist reviews have no photo attachments. photo_review_analysis() uses the product
   LISTING's photo count (products.product_photos_qty) as the closest available proxy.
5. ~0.07% of review rows have malformed date fields — the source CSV has review text with
   embedded newlines AND doubled-quote escaping (`""word""`) together, which Databricks'
   multiLine CSV reader doesn't fully resolve. Date-based tools filter these out via
   try_cast rather than crash or fabricate a date.
"""
import re
from collections import Counter

import numpy as np

from backend.db import DatabricksClient

_PT_STOPWORDS = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "não", "uma",
    "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "ao", "ele",
    "das", "seu", "sua", "ou", "quando", "muito", "nos", "já", "eu", "também", "só",
    "pelo", "pela", "até", "isso", "ela", "entre", "depois", "sem", "mesmo", "aos",
    "seus", "quem", "nas", "me", "esse", "eles", "você", "essa", "num", "nem",
    "suas", "meu", "às", "minha", "numa", "pelos", "elas", "qual", "nós", "lhe",
    "foi", "ser", "tem", "tinha", "está", "estava", "são", "for", "isto", "essas",
    "este", "esta", "estes", "todo", "toda", "todos", "todas", "tudo", "vc", "pra",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zà-ú]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _PT_STOPWORDS]


def search_reviews(db: DatabricksClient, query: str, limit: int = 10, min_score: float = 0.35) -> dict:
    """Semantic search over review text via the local multilingual RAG index."""
    from backend.rag import get_review_index

    index = get_review_index()
    matches = index.search(query, top_k=limit, min_score=min_score)
    return {
        "query": query,
        "matches": matches,
        "match_count": len(matches),
        "note": "semantic search via local multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2) - matches Portuguese review text regardless of the query's language",
    }


def sentiment_score_by_product(db: DatabricksClient, min_reviews: int = 5, top_n: int = 10) -> dict:
    """Average review_score (1-5, the dataset's direct sentiment signal) by product, top/bottom N."""
    sql = f"""
        SELECT
            p.product_id,
            COALESCE(t.product_category_name_english, p.product_category_name) AS category,
            ROUND(AVG(try_cast(r.review_score AS INT)), 2) AS avg_score,
            COUNT(*) AS review_count
        FROM {db.table('order_reviews')} r
        JOIN {db.table('orders')} o ON r.order_id = o.order_id
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        JOIN {db.table('products')} p ON oi.product_id = p.product_id
        LEFT JOIN {db.table('product_category_translation')} t ON p.product_category_name = t.product_category_name
        WHERE try_cast(r.review_score AS INT) IS NOT NULL
        GROUP BY p.product_id, category
        HAVING COUNT(*) >= {int(min_reviews)}
        ORDER BY avg_score DESC
    """
    rows = db.query(sql)
    return {
        "top_products": rows[:top_n],
        "bottom_products": rows[-top_n:][::-1] if len(rows) > top_n else [],
        "products_analyzed": len(rows),
        "min_reviews_threshold": min_reviews,
    }


def flag_negative_trend(db: DatabricksClient, months: int = 12) -> dict:
    """Monthly share of negative reviews (score <= 2), with trend direction."""
    sql = f"""
        SELECT
            date_trunc('month', try_cast(review_creation_date AS TIMESTAMP)) AS month,
            COUNT(*) AS total_reviews,
            SUM(CASE WHEN try_cast(review_score AS INT) <= 2 THEN 1 ELSE 0 END) AS negative_reviews
        FROM {db.table('order_reviews')}
        WHERE try_cast(review_creation_date AS TIMESTAMP) IS NOT NULL
          AND try_cast(review_score AS INT) IS NOT NULL
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT {int(months)}
    """
    rows = db.query(sql)
    rows.sort(key=lambda r: r["month"])
    for r in rows:
        r["negative_pct"] = round(int(r["negative_reviews"]) / int(r["total_reviews"]) * 100, 2) if int(r["total_reviews"]) else 0.0

    trend_direction = "flat"
    if len(rows) >= 2:
        delta = rows[-1]["negative_pct"] - rows[0]["negative_pct"]
        trend_direction = "worsening" if delta > 1 else "improving" if delta < -1 else "flat"

    return {"monthly": rows, "trend_direction": trend_direction}


def extract_common_complaints(db: DatabricksClient, min_score: int = 2, top_n: int = 15) -> dict:
    """Word-frequency proxy for complaint themes in negative reviews (raw Portuguese, no NLP layer yet)."""
    sql = f"""
        SELECT review_comment_message
        FROM {db.table('order_reviews')}
        WHERE try_cast(review_score AS INT) <= {int(min_score)}
          AND review_comment_message IS NOT NULL
          AND length(review_comment_message) > 0
    """
    rows = db.query(sql)
    counter = Counter()
    for r in rows:
        counter.update(_tokenize(r["review_comment_message"]))

    return {
        "reviews_analyzed": len(rows),
        "top_terms": [{"term": t, "count": c} for t, c in counter.most_common(top_n)],
        "note": "raw word-frequency on untranslated Portuguese text, not real NLP topic extraction",
    }


def review_response_time_correlation(db: DatabricksClient) -> dict:
    """Correlation between survey-answer latency (customer answering Olist's survey) and score.

    Not seller response time - Olist has no field for that.
    """
    sql = f"""
        SELECT
            try_cast(review_score AS INT) AS review_score,
            (unix_timestamp(review_answer_timestamp) - unix_timestamp(try_cast(review_creation_date AS TIMESTAMP))) / 3600.0 AS response_hours
        FROM {db.table('order_reviews')}
        WHERE review_answer_timestamp IS NOT NULL
          AND try_cast(review_creation_date AS TIMESTAMP) IS NOT NULL
          AND try_cast(review_score AS INT) IS NOT NULL
    """
    rows = db.query(sql)
    # Filter scores/hours together by the SAME row (not by truncating scores to len(hours)) -
    # a null response_hours can occur anywhere in the result set, not just at the end, and a
    # length-only truncation would silently misalign every score/hours pair after the first gap.
    paired = [
        (float(r["review_score"]), float(r["response_hours"]))
        for r in rows
        if r["response_hours"] is not None
    ]
    scores = np.array([p[0] for p in paired])
    hours = np.array([p[1] for p in paired])

    valid = hours >= 0
    scores, hours = scores[valid], hours[valid]
    correlation = float(np.corrcoef(scores, hours)[0, 1]) if len(hours) > 1 else 0.0

    return {
        "correlation": round(correlation, 3),
        "reviews_analyzed": len(hours),
        "mean_response_hours": round(float(hours.mean()), 1) if len(hours) else 0.0,
        "note": "measures time-to-answer Olist's satisfaction survey, not seller response time (no such field exists)",
    }


def sentiment_by_region(db: DatabricksClient) -> dict:
    """Average review_score by customer state."""
    sql = f"""
        SELECT
            c.customer_state AS state,
            ROUND(AVG(try_cast(r.review_score AS INT)), 2) AS avg_score,
            COUNT(*) AS review_count
        FROM {db.table('order_reviews')} r
        JOIN {db.table('orders')} o ON r.order_id = o.order_id
        JOIN {db.table('customers')} c ON o.customer_id = c.customer_id
        WHERE try_cast(r.review_score AS INT) IS NOT NULL
        GROUP BY c.customer_state
        HAVING COUNT(*) >= 10
        ORDER BY avg_score DESC
    """
    rows = db.query(sql)
    return {
        "by_state": rows,
        "best_state": rows[0] if rows else None,
        "worst_state": rows[-1] if rows else None,
    }


def photo_review_analysis(db: DatabricksClient) -> dict:
    """Correlation between product LISTING photo count and review score (reviews have no photos themselves)."""
    sql = f"""
        SELECT
            p.product_photos_qty,
            try_cast(r.review_score AS INT) AS review_score
        FROM {db.table('order_reviews')} r
        JOIN {db.table('orders')} o ON r.order_id = o.order_id
        JOIN {db.table('order_items')} oi ON o.order_id = oi.order_id
        JOIN {db.table('products')} p ON oi.product_id = p.product_id
        WHERE p.product_photos_qty IS NOT NULL
          AND try_cast(r.review_score AS INT) IS NOT NULL
    """
    rows = db.query(sql)
    photos = np.array([float(r["product_photos_qty"]) for r in rows])
    scores = np.array([float(r["review_score"]) for r in rows])
    correlation = float(np.corrcoef(photos, scores)[0, 1]) if len(rows) > 1 else 0.0

    buckets = {"1-2 photos": [], "3-5 photos": [], "6+ photos": []}
    for p, s in zip(photos, scores):
        key = "1-2 photos" if p <= 2 else "3-5 photos" if p <= 5 else "6+ photos"
        buckets[key].append(s)

    bucket_summary = {
        k: {"avg_score": round(float(np.mean(v)), 2), "count": len(v)}
        for k, v in buckets.items() if v
    }

    return {
        "correlation": round(correlation, 3),
        "reviews_analyzed": len(rows),
        "by_photo_bucket": bucket_summary,
        "note": "uses product listing photo count as a proxy - Olist reviews have no photo attachments",
    }
