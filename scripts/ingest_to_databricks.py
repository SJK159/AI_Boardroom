"""
Upload Olist CSVs to a Unity Catalog Volume, then create Delta tables via a one-time job.
"""
import os
import base64
from pathlib import Path
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

DATABRICKS_HOST = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
CATALOG = "workspace"
SCHEMA = "olist"
VOLUME = "raw_files"

CSV_TABLES = [
    ("olist_customers_dataset.csv",           "customers"),
    ("olist_orders_dataset.csv",              "orders"),
    ("olist_order_items_dataset.csv",         "order_items"),
    ("olist_order_payments_dataset.csv",      "order_payments"),
    ("olist_order_reviews_dataset.csv",       "order_reviews"),
    ("olist_products_dataset.csv",            "products"),
    ("olist_sellers_dataset.csv",             "sellers"),
    ("olist_geolocation_dataset.csv",         "geolocation"),
    ("product_category_name_translation.csv", "product_category_translation"),
]


def ensure_schema_and_volume(client: WorkspaceClient):
    print(f"\n=== Ensuring catalog.schema: {CATALOG}.{SCHEMA} ===")
    schemas = [s.name for s in client.schemas.list(catalog_name=CATALOG)]
    if SCHEMA not in schemas:
        client.schemas.create(name=SCHEMA, catalog_name=CATALOG)
        print(f"  Created schema: {SCHEMA}")
    else:
        print(f"  Schema exists: {SCHEMA}")

    print(f"=== Ensuring volume: {CATALOG}.{SCHEMA}.{VOLUME} ===")
    volumes = [v.name for v in client.volumes.list(catalog_name=CATALOG, schema_name=SCHEMA)]
    if VOLUME not in volumes:
        from databricks.sdk.service.catalog import VolumeType
        client.volumes.create(
            catalog_name=CATALOG,
            schema_name=SCHEMA,
            name=VOLUME,
            volume_type=VolumeType.MANAGED,
        )
        print(f"  Created volume: {VOLUME}")
    else:
        print(f"  Volume exists: {VOLUME}")


def upload_csvs(client: WorkspaceClient):
    print(f"\n=== Uploading CSVs to /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/ ===")
    for filename, _ in CSV_TABLES:
        local_path = DATA_DIR / filename
        volume_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{filename}"
        print(f"  Uploading {filename}...", end=" ", flush=True)
        with open(local_path, "rb") as f:
            client.files.upload(volume_path, f, overwrite=True)
        print("done")


NOTEBOOK_CONTENT = """\
# Databricks notebook source
# MAGIC %python

CATALOG = "workspace"
SCHEMA = "olist"
VOLUME = "raw_files"

tables = [
    ("olist_customers_dataset.csv",           "customers"),
    ("olist_orders_dataset.csv",              "orders"),
    ("olist_order_items_dataset.csv",         "order_items"),
    ("olist_order_payments_dataset.csv",      "order_payments"),
    ("olist_order_reviews_dataset.csv",       "order_reviews"),
    ("olist_products_dataset.csv",            "products"),
    ("olist_sellers_dataset.csv",             "sellers"),
    ("olist_geolocation_dataset.csv",         "geolocation"),
    ("product_category_name_translation.csv", "product_category_translation"),
]

for filename, table_name in tables:
    path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{filename}"
    print(f"Creating Delta table: {CATALOG}.{SCHEMA}.{table_name}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(path)
    df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{table_name}")
    print(f"  -> {table_name}: {df.count():,} rows")

print("\\nAll Delta tables created successfully.")
"""


def create_notebook(client: WorkspaceClient) -> str:
    nb_path = "/olist_ingestion"
    print(f"\n=== Creating notebook at {nb_path} ===")
    from databricks.sdk.service.workspace import ImportFormat, Language
    client.workspace.import_(
        path=nb_path,
        content=base64.b64encode(NOTEBOOK_CONTENT.encode()).decode(),
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        overwrite=True,
    )
    print("  Notebook created.")
    return nb_path


def create_delta_tables_via_sql(client: WorkspaceClient):
    """Use the serverless SQL warehouse to CREATE Delta tables from Volume CSVs."""
    from databricks.sdk.service.sql import StatementState

    warehouses = list(client.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouse found in this workspace.")
    warehouse_id = warehouses[0].id
    print(f"\n=== Using SQL warehouse: {warehouses[0].name} ({warehouse_id}) ===")

    statements = [f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}"]

    for filename, table_name in CSV_TABLES:
        vol_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{filename}"
        statements.append(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.{table_name}
USING DELTA
AS SELECT * FROM read_files(
  '{vol_path}',
  format => 'csv',
  header => true,
  inferSchema => true
)""")

    for stmt in statements:
        label = stmt.strip().split("\n")[0][:80]
        print(f"  Executing: {label}...", end=" ", flush=True)
        import time
        response = client.statement_execution.execute_statement(
            statement=stmt,
            warehouse_id=warehouse_id,
            wait_timeout="50s",
        )
        # Poll until terminal state if still pending
        while response.status and response.status.state in (
            StatementState.PENDING, StatementState.RUNNING
        ):
            time.sleep(5)
            response = client.statement_execution.get_statement(response.statement_id)

        state = response.status.state if response.status else StatementState.FAILED
        if state == StatementState.SUCCEEDED:
            print("done")
        else:
            error = response.status.error.message if response.status and response.status.error else "unknown"
            print(f"FAILED: {error}")
            raise RuntimeError(f"SQL statement failed: {error}")


if __name__ == "__main__":
    client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)
    ensure_schema_and_volume(client)
    upload_csvs(client)
    create_delta_tables_via_sql(client)
    print("\nIngestion complete. All tables live at workspace.olist.*")
