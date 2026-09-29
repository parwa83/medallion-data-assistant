"""Run the supplied Dobara medallion pipeline on local files with PySpark + Delta.

Expected input: CSV files in ./data/raw (override with --input).
Output: Delta tables under ./data/lake/{bronze,silver,gold}.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


REQUIRED_GOLD_INPUTS = (
    "raw_orders", "raw_order_items", "raw_products", "raw_stores",
    "raw_customers", "raw_inventory",
)


def make_spark(warehouse: Path) -> SparkSession:
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("Local Medallion Pipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", warehouse.resolve().as_uri())
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def table_name(path: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", path.stem).lower()
    if not name or name[0].isdigit():
        name = f"t_{name}"
    return name


def write_delta(df: DataFrame, path: Path) -> None:
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(str(path)))


def read_delta(path: Path) -> DataFrame:
    return SparkSession.getActiveSession().read.format("delta").load(str(path))


def clean_silver(df: DataFrame) -> DataFrame:
    for field in df.schema.fields:
        if field.dataType.simpleString() == "string" and not field.name.startswith("_"):
            df = df.withColumn(field.name, F.trim(F.col(field.name)))
    # Match the submitted full-row deduplication, including source-file lineage.
    return df.dropDuplicates(df.columns).drop("_ingest_timestamp").withColumn(
        "_silver_processed_timestamp", F.current_timestamp()
    )


def save_gold(df: DataFrame, path: Path, description: str) -> int:
    df = df.withColumn("_gold_processed_timestamp", F.current_timestamp())
    write_delta(df, path)
    count = df.count()
    print(f"  {path.stem}: {count:,} rows — {description}")
    return count


def build_gold(tables: dict[str, Path], out: Path) -> dict[str, int]:
    missing = [name for name in REQUIRED_GOLD_INPUTS if name not in tables]
    if missing:
        raise ValueError("Gold KPI tables require these CSVs in the raw input: " + ", ".join(missing))

    orders = read_delta(tables["raw_orders"]).withColumn(
        "order_status_norm", F.initcap(F.trim(F.lower("order_status")))
    )
    order_items = read_delta(tables["raw_order_items"])
    products = read_delta(tables["raw_products"])
    stores = read_delta(tables["raw_stores"])
    customers = read_delta(tables["raw_customers"])
    inventory = read_delta(tables["raw_inventory"])

    products = (products
        .withColumn("category_norm", F.initcap(F.trim(F.lower("category"))))
        .withColumn("sub_category_norm", F.initcap(F.trim(F.lower("sub_category"))))
        .withColumn("mrp_clean", F.regexp_extract("mrp", r"([0-9]+\.?[0-9]*)", 1).cast("double")))

    order_revenue = (
        orders.alias("o").join(order_items.alias("oi"), F.col("o.order_id") == F.col("oi.order_id"), "left")
        .withColumn("item_revenue", F.col("oi.item_price") * F.col("oi.quantity"))
        .groupBy(
            F.col("o.order_id"), F.to_date("o.order_timestamp").alias("order_date"),
            F.col("o.order_status_norm"), F.col("o.customer_id"), F.col("o.store_id"),
            F.col("o.payment_method"), F.col("o.delivery_fee"), F.col("o.delivery_time_minutes"),
        ).agg(F.sum("item_revenue").alias("order_revenue"),
              F.sum("oi.quantity").alias("total_items"), F.count("oi.item_id").alias("item_count"))
    )

    daily = (order_revenue.groupBy("order_date").agg(
        F.count("*").alias("total_orders"),
        F.sum(F.when(F.col("order_status_norm") == "Delivered", 1).otherwise(0)).alias("delivered_orders"),
        F.sum(F.when(F.col("order_status_norm") == "Cancelled", 1).otherwise(0)).alias("cancelled_orders"),
        F.sum(F.when(F.col("order_status_norm") == "In Transit", 1).otherwise(0)).alias("in_transit_orders"),
        F.sum(F.when(F.col("order_status_norm") == "Pending", 1).otherwise(0)).alias("pending_orders"),
        F.round(F.sum("order_revenue"), 2).alias("total_revenue"),
        F.round(F.sum("delivery_fee"), 2).alias("total_delivery_fee"),
        F.round(F.avg("order_revenue"), 2).alias("avg_order_value"),
        F.round(F.avg("delivery_time_minutes"), 2).alias("avg_delivery_time_min"),
        F.round(F.avg("total_items"), 1).alias("avg_items_per_order"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.countDistinct("store_id").alias("active_stores"),
    ).withColumn("cancel_rate_pct", F.round(F.col("cancelled_orders") / F.col("total_orders") * 100, 2))
      .withColumn("delivery_rate_pct", F.round(F.col("delivered_orders") / F.col("total_orders") * 100, 2))
      .orderBy("order_date"))

    customer_agg = order_revenue.groupBy("customer_id").agg(
        F.count("*").alias("total_orders_placed"),
        F.sum(F.when(F.col("order_status_norm") == "Delivered", 1).otherwise(0)).alias("delivered_orders"),
        F.sum(F.when(F.col("order_status_norm") == "Cancelled", 1).otherwise(0)).alias("cancelled_orders"),
        F.round(F.sum("order_revenue"), 2).alias("total_lifetime_spend"),
        F.round(F.avg("order_revenue"), 2).alias("avg_order_spend"),
        F.max("order_date").alias("last_order_date"), F.min("order_date").alias("first_order_date"))
    payment_window = Window.partitionBy("customer_id").orderBy(F.desc("method_count"), F.asc("payment_method"))
    preferred_payment = (orders.groupBy("customer_id", "payment_method").agg(F.count("*").alias("method_count"))
                         .withColumn("rn", F.row_number().over(payment_window)).where("rn = 1")
                         .select("customer_id", F.col("payment_method").alias("preferred_payment_method")))
    customer = (customers.alias("c").join(customer_agg.alias("ca"), "customer_id", "left")
        .join(preferred_payment.alias("pp"), "customer_id", "left")
        .select("customer_id", "full_name", "email", "city", "pincode", "device_type", "created_at",
            F.datediff(F.current_date(), F.to_date("created_at")).alias("account_tenure_days"),
            F.coalesce("total_orders_placed", F.lit(0)).alias("total_orders_placed"),
            F.coalesce("delivered_orders", F.lit(0)).alias("delivered_orders"),
            F.coalesce("cancelled_orders", F.lit(0)).alias("cancelled_orders"),
            F.coalesce("total_lifetime_spend", F.lit(0.0)).alias("total_lifetime_spend"),
            F.coalesce("avg_order_spend", F.lit(0.0)).alias("avg_order_spend"),
            F.coalesce("preferred_payment_method", F.lit("Unknown")).alias("preferred_payment_method"),
            "first_order_date", "last_order_date").orderBy(F.desc("total_lifetime_spend")))

    sales = (order_items.alias("oi").join(orders.alias("o"), F.col("oi.order_id") == F.col("o.order_id"), "left")
        .withColumn("is_delivered", F.when(F.col("o.order_status_norm") == "Delivered", 1).otherwise(0))
        .groupBy(F.col("oi.product_id")).agg(
            F.sum("oi.quantity").alias("total_quantity_sold"),
            F.round(F.sum(F.col("oi.item_price") * F.col("oi.quantity")), 2).alias("total_revenue"),
            F.round(F.avg("oi.item_price"), 2).alias("avg_selling_price"),
            F.countDistinct("o.order_id").alias("orders_containing_product"), F.sum("is_delivered").alias("delivered_units")))
    product = (products.alias("p").join(sales.alias("ps"), F.col("p.product_id") == F.col("ps.product_id"), "left")
        .select(F.col("p.product_id"), F.col("p.product_name"), F.col("category_norm").alias("category"),
            F.col("sub_category_norm").alias("sub_category"), F.col("p.unit"), F.col("mrp_clean").alias("mrp"),
            F.col("p.cost_price"), F.round(F.col("mrp_clean") - F.col("p.cost_price"), 2).alias("unit_margin"),
            F.coalesce("total_quantity_sold", F.lit(0)).alias("total_quantity_sold"),
            F.coalesce("total_revenue", F.lit(0.0)).alias("total_revenue"),
            F.coalesce("avg_selling_price", F.lit(0.0)).alias("avg_selling_price"),
            F.coalesce("orders_containing_product", F.lit(0)).alias("orders_containing_product"),
            F.coalesce("delivered_units", F.lit(0)).alias("delivered_units")).orderBy(F.desc("total_revenue")))

    store_agg = order_revenue.groupBy("store_id").agg(
        F.count("*").alias("total_orders"),
        F.sum(F.when(F.col("order_status_norm") == "Delivered", 1).otherwise(0)).alias("delivered_orders"),
        F.sum(F.when(F.col("order_status_norm") == "Cancelled", 1).otherwise(0)).alias("cancelled_orders"),
        F.round(F.sum("order_revenue"), 2).alias("total_revenue"), F.round(F.avg("order_revenue"), 2).alias("avg_order_value"),
        F.round(F.avg("delivery_time_minutes"), 2).alias("avg_delivery_time_min"), F.countDistinct("customer_id").alias("unique_customers_served"))
    low_stock = inventory.where(F.col("stock_on_hand") <= F.col("reorder_level")).groupBy("store_id").agg(F.count("*").alias("low_stock_items"))
    store = (stores.alias("s").join(store_agg.alias("sa"), "store_id", "left").join(low_stock.alias("ls"), "store_id", "left")
        .select("store_id", "store_name", "city", "zone", "operating_status", "opened_date",
            *[F.coalesce(c, F.lit(0)).alias(c) for c in ("total_orders", "delivered_orders", "cancelled_orders", "unique_customers_served", "low_stock_items")],
            *[F.coalesce(c, F.lit(0.0)).alias(c) for c in ("total_revenue", "avg_order_value", "avg_delivery_time_min")])
        .withColumn("cancel_rate_pct", F.round(F.col("cancelled_orders") / F.col("total_orders") * 100, 2))
        .orderBy(F.desc("total_revenue")))

    results = {}
    for name, frame, desc in (
        ("daily_revenue_metrics", daily, "one row per order date"),
        ("customer_360", customer, "one row per customer"),
        ("product_performance", product, "one row per product"),
        ("store_performance", store, "one row per store"),
    ):
        results[name] = save_gold(frame, out / "gold" / name, desc)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/Users/parwa/Desktop/PySpark/dataset"), help="Folder containing source CSV files")
    parser.add_argument("--output", type=Path, default=Path("data/lake"), help="Folder for local Delta tables")
    args = parser.parse_args()
    raw_dir, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    csv_files = sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.CSV"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")
    # Avoid processing duplicate paths on case-insensitive filesystems.
    csv_files = list(dict.fromkeys(csv_files))
    spark = make_spark(output / "_warehouse")
    try:
        bronze_tables: dict[str, Path] = {}
        print(f"Found {len(csv_files)} CSV files in {raw_dir}")
        for csv_path in csv_files:
            name = table_name(csv_path)
            target = output / "bronze" / name
            df = (spark.read.option("header", "true").option("inferSchema", "true")
                .option("escape", '"').option("multiLine", "true").csv(str(csv_path))
                .withColumn("_ingest_timestamp", F.current_timestamp())
                .withColumn("_source_file", F.lit(str(csv_path))))
            write_delta(df, target)
            bronze_tables[name] = target
            print(f"  Bronze {name}: {df.count():,} rows")

        silver_tables = {}
        print("\n=== SILVER ===")
        for name, bronze_path in bronze_tables.items():
            target = output / "silver" / name
            silver = clean_silver(read_delta(bronze_path))
            write_delta(silver, target)
            silver_tables[name] = target
            print(f"  Silver {name}: {silver.count():,} rows")

        print("\n=== GOLD KPIs (the second pasted script) ===")
        counts = build_gold(silver_tables, output)
        print("\n=== SUMMARY ===")
        for layer, layer_tables in (("BRONZE", bronze_tables), ("SILVER", silver_tables)):
            for name, path in layer_tables.items():
                print(f"{layer:7} {name:28} {read_delta(path).count():>12,} rows  {path}")
        for name, count in counts.items():
            print(f"GOLD    {name:28} {count:>12,} rows  {output / 'gold' / name}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
