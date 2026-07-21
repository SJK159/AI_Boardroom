from backend.agents.base_agent import SpecialistAgent
from backend.schemas import AgentType, Finding

from . import tools


class SentimentAgent(SpecialistAgent):
    agent_type = AgentType.SENTIMENT

    def analyze(self, query: str) -> list[Finding]:
        results = self._call_tools_parallel({
            "search_reviews": (tools.search_reviews, {"db": self.db, "query": query}),
            "sentiment_score_by_product": (tools.sentiment_score_by_product, {"db": self.db}),
            "flag_negative_trend": (tools.flag_negative_trend, {"db": self.db}),
            "extract_common_complaints": (tools.extract_common_complaints, {"db": self.db}),
            "review_response_time_correlation": (tools.review_response_time_correlation, {"db": self.db}),
            "sentiment_by_region": (tools.sentiment_by_region, {"db": self.db}),
            "photo_review_analysis": (tools.photo_review_analysis, {"db": self.db}),
        })
        search = results["search_reviews"]
        by_product = results["sentiment_score_by_product"]
        negative_trend = results["flag_negative_trend"]
        complaints = results["extract_common_complaints"]
        response_corr = results["review_response_time_correlation"]
        by_region = results["sentiment_by_region"]
        photo_analysis = results["photo_review_analysis"]

        findings = []

        findings.append(Finding(
            claim=f"Semantic search for '{query}' matched {search['match_count']} reviews above the relevance threshold",
            source="search_reviews",
            confidence=0.7,
            supporting_data=search,
            severity="info",
        ))

        worst = by_product["bottom_products"][0] if by_product["bottom_products"] else None
        findings.append(Finding(
            claim=f"Lowest-rated product (min {by_product['min_reviews_threshold']} reviews) is {worst['product_id']} at {worst['avg_score']}/5 ({worst['review_count']} reviews)" if worst else "No products meet the minimum review threshold",
            source="sentiment_score_by_product",
            confidence=0.85,
            supporting_data=by_product,
            severity="warning" if worst and float(worst["avg_score"]) < 2.5 else "info",
        ))

        findings.append(Finding(
            claim=f"Negative review share (score <=2) is {negative_trend['trend_direction']} across the last {len(negative_trend['monthly'])} months",
            source="flag_negative_trend",
            confidence=0.85,
            supporting_data=negative_trend,
            severity="warning" if negative_trend["trend_direction"] == "worsening" else "info",
        ))

        top_terms = ", ".join(t["term"] for t in complaints["top_terms"][:5])
        findings.append(Finding(
            claim=f"Top recurring terms in negative reviews (Portuguese, {complaints['reviews_analyzed']} reviews): {top_terms}" if complaints["top_terms"] else "No negative review text available to analyze",
            source="extract_common_complaints",
            confidence=0.5,
            supporting_data=complaints,
            severity="info",
        ))

        findings.append(Finding(
            claim=f"Survey-answer latency vs. review score correlation is {response_corr['correlation']} across {response_corr['reviews_analyzed']} reviews",
            source="review_response_time_correlation",
            confidence=0.6,
            supporting_data=response_corr,
            severity="info",
        ))

        best, worst_r = by_region["best_state"], by_region["worst_state"]
        findings.append(Finding(
            claim=f"Sentiment ranges from {best['avg_score']}/5 in {best['state']} to {worst_r['avg_score']}/5 in {worst_r['state']}" if best and worst_r else "Insufficient regional data",
            source="sentiment_by_region",
            confidence=0.8,
            supporting_data=by_region,
            severity="info",
        ))

        findings.append(Finding(
            claim=f"Product listing photo count vs. review score correlation is {photo_analysis['correlation']} across {photo_analysis['reviews_analyzed']} reviews",
            source="photo_review_analysis",
            confidence=0.55,
            supporting_data=photo_analysis,
            severity="info",
        ))

        return findings
