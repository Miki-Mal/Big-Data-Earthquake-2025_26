import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, avg, min, max, struct, lit, current_date, stddev, when
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import BisectingKMeans

def run_cluster_etl():
    spark = SparkSession.builder \
        .appName("Earthquake_ML_Clustering") \
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic") \
        .enableHiveSupport() \
        .getOrCreate()

    # --- Configuration ---
    SOURCE_TABLE = "earthquake_clean"
    DIM_TABLE = "Dim_Cluster_Metadata"
    FACT_TABLE = "Fact_Cluster_Daily_Stats"
    
    # 1. Load Data
    print("Loading data for ML Training...")
    # MODIFIED: Load ALL data instead of filtering by a specific date
    # This ensures we have enough points to form clusters.
    df_clean = spark.table(SOURCE_TABLE)
    
    # --- SAFETY CHECK ---
    # Check if the table is empty before proceeding
    row_count = df_clean.count()
    print(f"Total rows found for training: {row_count}")
    
    if row_count == 0:
        print("ERROR: Input table 'earthquake_clean' is empty! Cannot train model.")
        print("Please run the Clean ETL step first to populate data.")
        spark.stop()
        return

    # 2. Feature Engineering
    # Spark ML requires a vector column named 'features'
    # handle nulls by dropping them (ML fails on nulls)
    df_clean = df_clean.na.drop(subset=["latitude", "longitude"])

    assembler = VectorAssembler(
        inputCols=["latitude", "longitude"],
        outputCol="features"
    )
    df_vectorized = assembler.transform(df_clean)

    # 3. ML Task: Clustering (BisectingKMeans)
    # K=50 (approximate number of active seismic zones)
    print("Training BisectingKMeans Model...")
    bkm = BisectingKMeans().setK(50).setSeed(1).setFeaturesCol("features").setPredictionCol("cluster_id")
    
    # This is where it crashed before if data was empty
    model = bkm.fit(df_vectorized)

    # 4. Make Predictions (Assign Cluster IDs)
    df_clustered = model.transform(df_vectorized)

    # --- OUTPUT 1: Dim_Cluster_Metadata ---
    print(f"Generating {DIM_TABLE}...")
    
    dim_df = df_clustered.groupBy("cluster_id").agg(
        avg("latitude").alias("centroid_lat"),
        avg("longitude").alias("centroid_lon"),
        # Standard deviation provides a rough radius
        (stddev("latitude") + stddev("longitude")).alias("cluster_radius_deg"),
        count("*").alias("event_count_in_cluster")
    ).withColumn("creation_date", current_date().cast("string")) \
     .withColumn("cluster_radius_km", col("cluster_radius_deg") * 111.0) # Approx 111km per degree
    
    dim_final = dim_df.select(
        "cluster_id", 
        "centroid_lat", 
        "centroid_lon", 
        "cluster_radius_km", 
        "event_count_in_cluster",
        "creation_date"
    )

    dim_final.write.mode("overwrite").format("parquet").saveAsTable(DIM_TABLE)

    # --- OUTPUT 2: Fact_Cluster_Daily_Stats ---
    print(f"Generating {FACT_TABLE}...")
    
    fact_df = df_clustered.groupBy("dt", "cluster_id").agg(
        count("*").alias("total_events"),
        avg("magnitude").alias("avg_magnitude"),
        max("magnitude").alias("max_magnitude"),
        sum(when(col("magnitude") >= 6.0, 1).otherwise(0)).alias("mag_count_large"),
        avg("depth_km").alias("avg_depth"),
        min("depth_km").alias("depth_min"),
        max("depth_km").alias("depth_max"),
        max(struct(col("magnitude"), col("id"))).alias("max_mag_struct")
    )

    fact_final = fact_df.select(
        col("cluster_id"),
        col("total_events"),
        col("avg_magnitude"),
        col("max_magnitude"),
        col("mag_count_large"),
        col("avg_depth"),
        col("depth_min"),
        col("depth_max"),
        col("max_mag_struct.id").alias("max_mag_event_id"),
        col("dt")
    )

    fact_final.write \
        .mode("overwrite") \
        .partitionBy("dt") \
        .format("parquet") \
        .saveAsTable(FACT_TABLE)

    print("Success: ML Clustering and ETL complete.")
    spark.stop()

if __name__ == "__main__":
    run_cluster_etl()