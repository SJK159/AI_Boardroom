"""Run the Finance Agent against live Delta tables and print its briefing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.agents.finance import FinanceAgent
from backend.db import DatabricksClient

if __name__ == "__main__":
    db = DatabricksClient()
    agent = FinanceAgent(db)
    briefing = agent.run("How is our financial health looking?")

    print(f"Agent: {briefing.agent.value}")
    print(f"Execution time: {briefing.execution_time_ms:.0f}ms")
    print(f"Tool calls: {len(briefing.tool_calls)} ({sum(1 for t in briefing.tool_calls if t.success)} succeeded)\n")

    for f in briefing.findings:
        print(f"[{f.severity.upper()}] ({f.confidence:.2f}) {f.claim}")
        print(f"  source: {f.source}\n")
