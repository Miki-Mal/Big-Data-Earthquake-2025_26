import math
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, DateType, LongType

# Start Total Timer
total_start_time = time.time()

# 1. Initialise Spark Session
spark = SparkSession.builder \
    .appName("Earthquake_Clustering_Final_Pipeline") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2. Load Data
# Filter out null coordinates for clustering
df_full = spark.table("earthquake_clean")

df_raw = df_full.filter("latitude IS NOT NULL AND longitude IS NOT NULL") \
    .select("id", "latitude", "longitude")

# 3. Feature Engineering: Spherical to Cartesian
df_feat = df_raw.withColumn("lat_rad", F.radians(F.col("latitude"))) \
                .withColumn("lon_rad", F.radians(F.col("longitude"))) \
                .withColumn("x", F.cos(F.col("lat_rad")) * F.cos(F.col("lon_rad"))) \
                .withColumn("y", F.cos(F.col("lat_rad")) * F.sin(F.col("lon_rad"))) \
                .withColumn("z", F.sin(F.col("lat_rad")))

assembler = VectorAssembler(inputCols=["x", "y", "z"], outputCol="features")
df_vector = assembler.transform(df_feat).select("id", "features", "x", "y", "z").cache()

# 4. Calculate TSS
N = df_vector.count()
global_means = df_vector.select(F.mean("x"), F.mean("y"), F.mean("z")).collect()[0]
mean_x, mean_y, mean_z = global_means[0], global_means[1], global_means[2]

df_tss = df_vector.withColumn("sq_dist_global", 
    F.pow(F.col("x") - mean_x, 2) + 
    F.pow(F.col("y") - mean_y, 2) + 
    F.pow(F.col("z") - mean_z, 2)
)
tss = df_tss.select(F.sum("sq_dist_global")).collect()[0][0]

print(f"Data Loaded. N={N}, TSS={tss:.4f}")

# 5. Iterative Clustering
results = []
k_range = range(300, 601, 20)
num_trials = 3

global_best_score = -1.0
global_best_k = -1
global_best_model = None

print("\nStarting Clustering Loop...")
print(f"{'K':<5} | {'Trial':<5} | {'Time(s)':<8} | {'WCSS':<15} | {'CH Index':<15}")
print("-" * 60)

for k in k_range:
    for trial in range(num_trials):
        trial_start = time.time()
        
        seed = 42 + trial + (k * 10)
        kmeans = KMeans().setK(k).setSeed(seed).setFeaturesCol("features").setInitMode("k-means||") 
        model = kmeans.fit(df_vector)
        
        wcss = model.summary.trainingCost
        
        if wcss == 0 or N <= k:
            ch_index = 0.0
        else:
            ssb = tss - wcss
            ch_index = (ssb / (k - 1)) / (wcss / (N - k))
        
        trial_duration = time.time() - trial_start
        print(f"{k:<5} | {trial:<5} | {trial_duration:<8.2f} | {wcss:<15.2f} | {ch_index:<15.2f}")
        
        results.append((int(k), int(trial), float(wcss), float(ch_index), float(trial_duration)))
        
        if ch_index > global_best_score:
            global_best_score = ch_index
            global_best_k = k
            global_best_model = model

print("-" * 60)
print(f"Global Best Model: K={global_best_k}, CH Index={global_best_score:.2f}")

# 6. Save Metrics (Local)
schema_metrics = StructType([
    StructField("k", IntegerType(), False),
    StructField("trial", IntegerType(), False),
    StructField("wcss", DoubleType(), False),
    StructField("ch_index", DoubleType(), False),
    StructField("execution_time_sec", DoubleType(), False)
])
spark.createDataFrame(results, schema_metrics).coalesce(1).write.mode("overwrite") \
    .option("header", "true").csv("file:///home/vagrant/earthquake_clustering_metrics")

# 7. Extract Cluster Metadata (HDFS)
# Calculate Max Radius for visualization
print("Calculating Metadata and Max Radius...")
df_pred = global_best_model.transform(df_vector).select("id", "prediction")
centers = global_best_model.clusterCenters()

centers_map = {}
for i, center in enumerate(centers):
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    norm = math.sqrt(cx*cx + cy*cy + cz*cz)
    if norm == 0: norm = 1
    centers_map[i] = (cx/norm, cy/norm, cz/norm)

R_EARTH = 6371.0 
def get_dist(cluster_id, x, y, z):
    if cluster_id not in centers_map: return 0.0
    cx, cy, cz = centers_map[cluster_id]
    dot_prod = max(-1.0, min(1.0, x*cx + y*cy + z*cz))
    return math.acos(dot_prod) * R_EARTH

dist_udf = F.udf(get_dist, DoubleType())

# We join back to df_vector to get X/Y/Z for distance calc
df_dist_calc = df_pred.join(df_vector.select("id", "x", "y", "z"), "id") \
    .withColumn("dist_km", dist_udf(F.col("prediction"), F.col("x"), F.col("y"), F.col("z")))

max_r_rows = df_dist_calc.groupBy("prediction").agg(F.max("dist_km").alias("max_r")).collect()
max_r_map = {row['prediction']: row['max_r'] for row in max_r_rows}

metadata_rows = []
for i, center in enumerate(centers):
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    R = math.sqrt(cx**2 + cy**2 + cz**2)
    if R == 0: lat, lon = 0.0, 0.0
    else:
        lat = math.degrees(math.asin(cz/R))
        lon = math.degrees(math.atan2(cy, cx))
    
    metadata_rows.append((i, cx, cy, cz, lat, lon, max_r_map.get(i, 0.0)))

schema_meta = StructType([
    StructField("cluster_id", IntegerType(), False),
    StructField("center_x", DoubleType(), False),
    StructField("center_y", DoubleType(), False),
    StructField("center_z", DoubleType(), False),
    StructField("center_lat", DoubleType(), False),
    StructField("center_lon", DoubleType(), False),
    StructField("max_r_km", DoubleType(), False)
])
spark.createDataFrame(metadata_rows, schema_meta).write.mode("overwrite") \
    .option("header", "true").csv("/user/testuser/earthquake_data/dim_cluster_metadata")

# -------------------------------------------------------------
# 8. Aggregation: Daily Stats per Cluster (HDFS - PARQUET)
# -------------------------------------------------------------
print("Aggregating Daily Statistics per Cluster...")

# Join predictions with full data
df_joined = df_full.join(df_pred, "id")

# Aggregation Logic
df_agg = df_joined.groupBy(F.col("prediction").alias("cluster_id"), F.col("dt")) \
    .agg(
        F.count("*").alias("total_events"),
        
        # --- Magnitude Buckets (Aligned with Regions Script) ---
        # Small (< 4.0), Medium (4.0 - 5.9), Large (>= 6.0)
        F.sum(F.when(F.col("magnitude") < 4.0, 1).otherwise(0)).alias("mag_count_small"),
        F.sum(F.when((F.col("magnitude") >= 4.0) & (F.col("magnitude") < 6.0), 1).otherwise(0)).alias("mag_count_medium"),
        F.sum(F.when(F.col("magnitude") >= 6.0, 1).otherwise(0)).alias("mag_count_large"),
        
        # --- Magnitude Statistics ---
        F.sum("magnitude").alias("sum_magnitude"),
        F.avg("magnitude").alias("avg_magnitude"),
        F.max("magnitude").alias("max_magnitude"),
        
        # --- Depth Statistics ---
        F.sum("depth_km").alias("depth_sum"),
        F.avg("depth_km").alias("avg_depth"),
        F.min("depth_km").alias("depth_min"),
        F.max("depth_km").alias("depth_max"),
        
        # --- Biggest Earthquake Info ---
        # Struct captures the row with max magnitude to extract details later
        F.max(F.struct(
            F.col("magnitude"), 
            F.col("id"), 
            F.col("longitude"), 
            F.col("latitude"), 
            F.col("depth_km"), 
            F.col("place")
        )).alias("max_event_struct")
    )

# Unpack the struct columns and select in the exact requested order
df_final_agg = df_agg.select(
    F.col("cluster_id"),
    F.col("total_events"),
    F.col("mag_count_small"),
    F.col("mag_count_medium"),
    F.col("mag_count_large"),
    F.col("sum_magnitude"),
    F.col("avg_magnitude"),
    F.col("max_magnitude"),
    F.col("depth_sum"),
    F.col("avg_depth"),
    F.col("depth_min"),
    F.col("depth_max"),
    F.col("max_event_struct.id").alias("max_mag_event_id"),
    F.col("max_event_struct.longitude").alias("max_mag_event_longitude"),
    F.col("max_event_struct.latitude").alias("max_mag_event_latitude"),
    F.col("max_event_struct.depth_km").alias("max_mag_event_depth_km"),
    F.col("max_event_struct.place").alias("max_mag_event_place"),
    F.col("dt")
)

# Save to HDFS as PARQUET
agg_path = "/user/testuser/earthquake_data/fact_cluster_daily_stats"
print(f"Saving daily aggregations (Parquet) to {agg_path}...")

# Use coalesce(1) to prevent "Small File Problem"
df_final_agg.coalesce(1).write.mode("overwrite").parquet(agg_path)

# -------------------------------------------------------------

total_duration = time.time() - total_start_time
print("-" * 60)
print("Process Complete.")
print(f"Total Execution Time: {total_duration:.2f} seconds")

spark.stop()