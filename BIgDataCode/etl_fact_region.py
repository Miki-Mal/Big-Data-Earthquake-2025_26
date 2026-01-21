import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, broadcast, count, sum, avg, min, max, struct, when, lit
)
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType

def run_regional_aggregation():
    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("Fact_Region_Daily_Stats_ETL_Dec2025") \
        .enableHiveSupport() \
        .getOrCreate()

    # 2. Define Paths
    SOURCE_TABLE = "earthquake_clean"
    # DIRECT HDFS PATH to the CSV files (Bypassing Hive Table issues)
    REGION_HDFS_PATH = "/user/testuser/earthquake_data/dim_region_definitions_dir"
    TARGET_PATH = "/user/testuser/earthquake_data/region_daily_analytics"

    # 3. Load Region Definitions (Directly from HDFS)
    print(f"Loading regions from HDFS path: {REGION_HDFS_PATH}")
    
    # We define the schema manually to ensure types are correct immediately
    region_schema = StructType([
        StructField("region_id", IntegerType(), True),
        StructField("region", StringType(), True),
        StructField("sub_region", StringType(), True),
        StructField("lat_min", DoubleType(), True),
        StructField("lat_max", DoubleType(), True),
        StructField("lon_min", DoubleType(), True),
        StructField("lon_max", DoubleType(), True)
    ])

    # Read CSV directly. We assume header exists based on your Hive DDL.
    regions_df = spark.read \
        .option("header", "true") \
        .schema(region_schema) \
        .csv(REGION_HDFS_PATH)

    # 4. Load Earthquake Data (Filtered for Dec 2025)
    print(f"Loading data from table: {SOURCE_TABLE}")
    eq_df = spark.table(SOURCE_TABLE).filter(col("dt") >= "2025-12-01")

    # Debug Counts
    print(f"Earthquake rows (Dec 2025): {eq_df.count()}")
    print(f"Region rows: {regions_df.count()}")
    regions_df.show(5, truncate=False) # Verify we loaded regions correctly

    # 5. Spatial Join (Broadcast)
    joined_df = eq_df.join(
        broadcast(regions_df),
        (eq_df.latitude >= regions_df.lat_min) & 
        (eq_df.latitude <= regions_df.lat_max) & 
        (eq_df.longitude >= regions_df.lon_min) & 
        (eq_df.longitude <= regions_df.lon_max)
    )

    # 6. Aggregation Logic
    agg_df = joined_df.groupBy("dt", "region_id", "region", "sub_region").agg(
        count("*").alias("total_events"),
        sum(when(col("magnitude") < 4.0, 1).otherwise(0)).alias("mag_count_small"),
        sum(when((col("magnitude") >= 4.0) & (col("magnitude") < 6.0), 1).otherwise(0)).alias("mag_count_medium"),
        sum(when(col("magnitude") >= 6.0, 1).otherwise(0)).alias("mag_count_large"),
        sum("magnitude").alias("sum_magnitude"),
        avg("magnitude").alias("avg_magnitude"),
        max("magnitude").alias("max_magnitude"),
        sum("depth_km").alias("depth_sum"),
        avg("depth_km").alias("avg_depth"),
        min("depth_km").alias("depth_min"),
        max("depth_km").alias("depth_max"),
        max(struct(col("magnitude"), col("id"), col("place"))).alias("max_mag_info")
    )

    # 7. Final Formatting
    final_df = agg_df.select(
        col("dt"),
        col("region_id"),
        col("region"),
        col("sub_region"), 
        col("total_events"),
        col("mag_count_small"),
        col("mag_count_medium"),
        col("mag_count_large"),
        col("sum_magnitude"),
        col("avg_magnitude"),
        col("max_magnitude"),
        col("depth_sum"),
        col("avg_depth"),
        col("depth_min"),
        col("depth_max"),
        col("max_mag_info.id").alias("max_mag_event_id"),
        col("max_mag_info.place").alias("max_mag_event_place")
    )

    # 8. Write Strategy
    print(f"Writing aggregated data to: {TARGET_PATH}")
    
    if final_df.take(1) == []:
        print("WARNING: The DataFrame is empty! No earthquakes matched the regions.")
    else:
        final_df.coalesce(1).write \
            .mode("overwrite") \
            .parquet(TARGET_PATH)
        print("Success: Aggregation complete.")

    spark.stop()

if __name__ == "__main__":
    run_regional_aggregation()