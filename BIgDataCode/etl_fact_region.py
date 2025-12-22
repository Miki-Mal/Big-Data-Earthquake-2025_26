import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, broadcast, count, sum, avg, min, max, struct, when, lit
)

def run_regional_aggregation():
    # 1. Initialize Spark Session with Hive Support
    spark = SparkSession.builder \
        .appName("Fact_Region_Daily_Stats_ETL") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .enableHiveSupport() \
        .getOrCreate()

    # 2. Define File Paths and Table Names
    REGION_CSV_PATH = "file:///home/vagrant/earthquake_redion_definitions/region_definitions.csv"
    SOURCE_TABLE = "earthquake_clean"
    TARGET_TABLE = "Fact_Region_Daily_Stats"

    # 3. Load Region Definitions
    # We infer schema to ensure lat/lon are doubles
    print(f"Loading regions from: {REGION_CSV_PATH}")
    regions_df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(REGION_CSV_PATH)

    # 4. Load Cleaned Earthquake Data
    # We read from the Silver Layer table
    print(f"Loading data from table: {SOURCE_TABLE}")
    eq_df = spark.table(SOURCE_TABLE)

    # 5. Spatial Join (Broadcast Strategy)
    # Since regions_df is tiny (~16 rows), we broadcast it to all nodes.
    # This avoids a massive Cartesian product and makes the join instant.
    joined_df = eq_df.join(
        broadcast(regions_df),
        (eq_df.latitude >= regions_df.lat_min) & 
        (eq_df.latitude <= regions_df.lat_max) & 
        (eq_df.longitude >= regions_df.lon_min) & 
        (eq_df.longitude <= regions_df.lon_max)
    )

    # 6. Aggregation Logic
    # Group by Date and Region to calculate daily statistics
    agg_df = joined_df.groupBy("dt", "region_id").agg(
        # --- Event Counts ---
        count("*").alias("total_events"),
        
        # --- Magnitude Buckets ---
        # Small (< 4.0), Medium (4.0 - 5.9), Large (>= 6.0)
        sum(when(col("magnitude") < 4.0, 1).otherwise(0)).alias("mag_count_small"),
        sum(when((col("magnitude") >= 4.0) & (col("magnitude") < 6.0), 1).otherwise(0)).alias("mag_count_medium"),
        sum(when(col("magnitude") >= 6.0, 1).otherwise(0)).alias("mag_count_large"),
        
        # --- Magnitude Statistics ---
        sum("magnitude").alias("sum_magnitude"), # Useful for weighted avgs if needed later
        avg("magnitude").alias("avg_magnitude"),
        max("magnitude").alias("max_magnitude"),
        
        # --- Depth Statistics ---
        sum("depth_km").alias("depth_sum"),
        avg("depth_km").alias("avg_depth"),
        min("depth_km").alias("depth_min"),
        max("depth_km").alias("depth_max"),
        
        # --- Biggest Earthquake Info (The Struct Trick) ---
        # Finds the row with the max magnitude and extracts its ID and Place
        max(struct(col("magnitude"), col("id"), col("place"))).alias("max_mag_info")
    )

    # 7. Final Formatting
    final_df = agg_df.select(
        col("dt"),
        col("region_id"),
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

    # 8. Write to Hive Table (Gold Layer)
    # coalesce(1) prevents the "Small File Problem" since daily stats are tiny rows.
    print(f"Writing aggregated data to: {TARGET_TABLE}")
    final_df.coalesce(1).write \
        .mode("overwrite") \
        .partitionBy("dt") \
        .format("parquet") \
        .saveAsTable(TARGET_TABLE)

    print("Success: Aggregation complete.")
    spark.stop()

if __name__ == "__main__":
    run_regional_aggregation()