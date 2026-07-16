from enum import Enum


class AgentType(str, Enum):
    FINANCE = "Finance"
    SALES = "Sales"
    SENTIMENT = "Sentiment"
    OPERATIONS = "Operations"
    GROWTH = "Growth"
    RISK = "Risk"
    COMPLIANCE = "Compliance"
