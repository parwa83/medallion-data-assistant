# Run the medallion pipeline locally in VS Code

This is the local PySpark + Delta Lake version of the two supplied Databricks scripts. It writes Delta tables as folders on your machine rather than creating Unity Catalog tables. The second pasted script's four business KPI tables are the final Gold layer; the earlier generic one-row aggregate Gold tables are superseded.

## 1. Prerequisites

- Python 3.9 or newer
- Java 17 or newer, with `JAVA_HOME` set
- VS Code with the Python extension
- Internet access for the first install/run so Delta Lake can download its Spark JVM package

The pinned versions below are a compatible Delta Lake 4.0.0 / Spark 4.0.0 pair. See the [Delta Lake quick start](https://docs.delta.io/quick-start/) and [PySpark installation guide](https://spark.apache.org/docs/latest/api/python/getting_started/install.html).

## 2. Prepare the project

Open this folder in VS Code. Put the input CSV files in `data/raw/`. The KPI step expects these filenames (case-insensitive extension; names are normalized to lowercase):

```text
raw_orders.csv
raw_order_items.csv
raw_products.csv
raw_stores.csv
raw_customers.csv
raw_inventory.csv
```

Create a virtual environment and install dependencies from the VS Code terminal:

```bash
python -m venv .venv
```

Activate it, then run:

```bash
python -m pip install -r requirements.txt
python medallion_local.py
```

On macOS/Linux, activate with `source .venv/bin/activate`; on Windows PowerShell, use `.venv\Scripts\Activate.ps1`. Select that `.venv` interpreter in VS Code.

To use different folders:

```bash
python medallion_local.py --input /path/to/csvs --output /path/to/local/lake
```

## 3. Outputs

The pipeline creates Delta data under `data/lake/`:

```text
data/lake/bronze/<source_name>/
data/lake/silver/<source_name>/
data/lake/gold/daily_revenue_metrics/
data/lake/gold/customer_360/
data/lake/gold/product_performance/
data/lake/gold/store_performance/
```

Reruns overwrite these local outputs, matching the supplied scripts' overwrite behavior. To query a table from another PySpark session configured for Delta, read the folder with `spark.read.format("delta").load("data/lake/gold/customer_360")`.

## Local differences from Databricks

- `/Volumes/...`, `dbutils.fs.ls`, Unity Catalog schemas, `saveAsTable`, and managed-table properties are replaced by local filesystem discovery and Delta folders.
- Change Data Feed settings, `OPTIMIZE`, table comments, and Databricks data-skipping properties are not included. Those depend on Databricks/Delta table management; local Delta files remain queryable.
- Gold KPI logic is retained. Preferred payment is computed by the most frequently used method, with a deterministic tie-break. The pasted version sorted by frequency but then chose the lexicographically greatest payment name, which did not reliably select the most frequent method.
- The output uses overwrite mode, so each run rebuilds the layers from the current CSV files; it is not an incremental streaming pipeline.



### To acrtivate:
### parwa@Parwas-MacBook-Air PySpark % source .venv/bin/activate


### To ask question
### ((.venv) ) parwa@Parwas-MacBook-Air PySpark % python rag_gold.py --chat-model openai/gpt-oss-120b:fastest ask "Which customer had spend the most?"