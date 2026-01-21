# Code was written with the help of AI

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, row_number, lit, floor, to_timestamp
)

def run_etl():
    spark = SparkSession.builder \
        .appName("Earthquake_Unified_Clean") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .enableHiveSupport() \
        .getOrCreate()
        
    # --- 0. Register new partitions ---
    # Hive needs to be refreshed
    spark.sql("MSCK REPAIR TABLE usgs_raw")
    spark.sql("MSCK REPAIR TABLE terraquake_raw")

    # --- 1. Load Data from Hive Tables ---
    # Using standard table names based on architecture
    # Path to raw data folders
    usgs_path = "/user/testuser/earthquake_data/USGS_raw/"
    terra_path = "/user/testuser/earthquake_data/terraquake_raw/"

    df_usgs_raw = spark.read.option("basePath", usgs_path).parquet(usgs_path)
    df_terra_raw = spark.read.option("basePath", terra_path).parquet(terra_path)

    # --- 2. Transform USGS to Unified Schema ---
    df_usgs_unified = df_usgs_raw.select(
        col("id"),
        col("dt"),
        col("properties.mag").cast("double").alias("magnitude"),
        lit("ml").alias("mag_type"), # magType is always "ml" for USGS data
        col("properties.place").alias("place"),
        col("properties.time").cast("bigint").alias("time_utc"),
        col("geometry.coordinates").getItem(0).cast("double").alias("longitude"),
        col("geometry.coordinates").getItem(1).cast("double").alias("latitude"),
        col("geometry.coordinates").getItem(2).cast("double").alias("depth_km"),
        lit("USGS").alias("source_system") # Tag source for prioritization later
    )

    # --- 3. Transform Terraquake to Unified Schema ---
    df_terra_unified = df_terra_raw.select(
        col("properties.eventId").cast("string").alias("id"),
        col("dt"),
        col("properties.mag").cast("double").alias("magnitude"),
        col("properties.magType").alias("mag_type"),
        col("properties.place").alias("place"),
        to_timestamp(col("properties.time")).cast("long").alias("time_utc"),
        col("geometry.coordinates").getItem(0).cast("double").alias("longitude"),
        col("geometry.coordinates").getItem(1).cast("double").alias("latitude"),
        col("geometry.coordinates").getItem(2).cast("double").alias("depth_km"),
        lit("Terraquake").alias("source_system")
    )

    # --- 4. Union the sources ---
    df_combined = df_usgs_unified.unionByName(df_terra_unified)

    # --- 5. Exact Deduplication (by ID) ---
    # If the same ID appears twice, keep the one with the latest timestamp info
    window_id = Window.partitionBy("id").orderBy(col("time_utc").desc())
    
    df_dedup_id = df_combined.withColumn("rn", row_number().over(window_id)) \
        .filter(col("rn") == 1) \
        .drop("rn")

    # --- 6. Fuzzy Deduplication (Spatial-Temporal) ---
    # Problem: Earthquake have different IDs in USGS vs Terraquake.
    # Solution: Create "Buckets" to identify events that are physically the same.
    
    # Logic:
    # 1. Time Bucket: Divide time by 60000ms (1 minute). Events in the same minute match.
    # 2. Geo Bucket: Multiply lat/lon by 10 and floor. Matches events within ~11km (0.1 deg).
    
    df_fuzzy_prep = df_dedup_id.withColumn("time_bucket", floor(col("time_utc") / 60000)) \
                               .withColumn("lat_bucket", floor(col("latitude") * 10)) \
                               .withColumn("lon_bucket", floor(col("longitude") * 10))

    # Window partition by these loose buckets.
    # Order By:
    # 1. source_system DESC -> 'USGS' comes after 'Terraquake' alphabetically, 
    #    so DESC puts USGS first. We trust USGS more.
    # 2. magnitude DESC -> keep the higher magnitude reading if sources are equal.
    window_fuzzy = Window.partitionBy("time_bucket", "lat_bucket", "lon_bucket") \
                         .orderBy(col("source_system").desc(), col("magnitude").desc())

    df_final_clean = df_fuzzy_prep.withColumn("rn_fuzzy", row_number().over(window_fuzzy)) \
        .filter(col("rn_fuzzy") == 1) \
        .drop("rn_fuzzy", "time_bucket", "lat_bucket", "lon_bucket")

    # --- 7. Write to Storage ---
    clean_table_name = "earthquake_clean"
    output_path = "/user/testuser/earthquake_data/clean"

    df_final_clean.coalesce(1).write \
        .mode("overwrite") \
        .partitionBy("dt") \
        .format("parquet") \
        .option("path", output_path) \
        .saveAsTable(clean_table_name)
        
    #print(f"Success: Cleaned data written to Hive table '{clean_table_name}'")
    spark.stop()

if __name__ == "__main__":
    run_etl()