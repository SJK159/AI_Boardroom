import time
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

from backend.config import settings


class DatabricksClient:
    """Thin wrapper for running SQL against Delta tables via the SQL warehouse.

    Specialist agent tools call `query()` — no agent code touches the SDK directly,
    so swapping compute (warehouse vs. cluster) later only touches this file.
    """

    def __init__(self):
        self._client = WorkspaceClient(
            host=settings.databricks_host, token=settings.databricks_token
        )
        self._warehouse_id = settings.databricks_warehouse_id or self._first_warehouse_id()

    def _first_warehouse_id(self) -> str:
        warehouses = list(self._client.warehouses.list())
        if not warehouses:
            raise RuntimeError("No SQL warehouse available in this workspace.")
        return warehouses[0].id

    def query(self, sql: str, poll_interval_s: float = 2.0) -> list[dict[str, Any]]:
        """Execute a SQL statement and return rows as a list of dicts."""
        response = self._client.statement_execution.execute_statement(
            statement=sql,
            warehouse_id=self._warehouse_id,
            wait_timeout="50s",
        )

        while response.status and response.status.state in (
            StatementState.PENDING,
            StatementState.RUNNING,
        ):
            time.sleep(poll_interval_s)
            response = self._client.statement_execution.get_statement(response.statement_id)

        if response.status and response.status.state != StatementState.SUCCEEDED:
            error = response.status.error.message if response.status.error else "unknown error"
            raise RuntimeError(f"Query failed: {error}\nSQL: {sql}")

        if not response.result or not response.manifest:
            return []

        columns = [col.name for col in response.manifest.schema.columns]
        rows = response.result.data_array or []
        return [dict(zip(columns, row)) for row in rows]

    def table(self, table_name: str) -> str:
        """Fully-qualified table name: catalog.schema.table"""
        return f"{settings.databricks_catalog}.{settings.databricks_schema}.{table_name}"
