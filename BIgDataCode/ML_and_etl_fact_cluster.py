import math
import time
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import BisectingKMeans
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType

# Start Total Timer
total_start_time = time.time()

# 1. Initialise Spark Session
spark = SparkSession.builder \
    .appName("Earthquake_Clustering_Dec2025_Only") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# 2. Load Data & FILTER FOR DEC 2025
print("Loading data (December 2025 only)...")
df_full = spark.table("earthquake_clean").filter(F.col("dt") >= "2025-12-01") # <--- NEW FILTER

df_raw = df_full.filter("latitude IS NOT NULL AND longitude IS NOT NULL") \
    .select("id", "latitude", "longitude")

row_count = df_raw.count()
print(f"Data Loaded. Rows for clustering: {row_count}")

if row_count == 0:
    print("ERROR: No data found for Dec 2025. Exiting.")
    spark.stop()
    sys.exit(1)

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

print(f"TSS calculated: {tss:.4f}")

# 5. Iterative Clustering (Reduced K Range for Testing)
results = []
# REDUCED K-RANGE for faster testing since data is smaller
k_range = range(50, 350, 50) 
num_trials = 2

global_best_score = -1.0
global_best_k = -1
global_best_model = None

print("\n" + "="*80)
print(f"{'K':<6} | {'Trial':<6} | {'Time(s)':<10} | {'WCSS':<15} | {'CH Index':<15}")
print("-" * 80)

for k in k_range:
    for trial in range(num_trials):
        trial_start = time.time()
        seed = 42 + trial + (k * 10)
        
        bkmeans = BisectingKMeans() \
            .setK(k) \
            .setSeed(seed) \
            .setFeaturesCol("features") \
            .setMinDivisibleClusterSize(1.0)
            
        model = bkmeans.fit(df_vector)
        wcss = model.summary.trainingCost
        
        if wcss == 0 or N <= k:
            ch_index = 0.0
        else:
            ssb = tss - wcss
            ch_index = (ssb / (k - 1)) / (wcss / (N - k))
        
        trial_duration = time.time() - trial_start
        print(f"{k:<6} | {trial:<6} | {trial_duration:<10.2f} | {wcss:<15.2f} | {ch_index:<15.2f}")
        sys.stdout.flush()
        
        if ch_index > global_best_score:
            global_best_score = ch_index
            global_best_k = k
            global_best_model = model

print("=" * 80)
print(f"Winner: K={global_best_k}")

# 6. Extract Metadata
print("\nExtracting centroids...")
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
df_metadata = spark.createDataFrame(metadata_rows, schema_meta)

# 7. Aggregation & Denormalization
print("Aggregating daily stats and denormalizing...")
df_joined = df_full.join(df_pred, "id")

df_agg = df_joined.groupBy(F.col("prediction").alias("cluster_id"), F.col("dt")) \
    .agg(
        F.count("*").alias("total_events"),
        F.sum(F.when(F.col("magnitude") < 4.0, 1).otherwise(0)).alias("mag_count_small"),
        F.sum(F.when((F.col("magnitude") >= 4.0) & (F.col("magnitude") < 6.0), 1).otherwise(0)).alias("mag_count_medium"),
        F.sum(F.when(F.col("magnitude") >= 6.0, 1).otherwise(0)).alias("mag_count_large"),
        F.sum("magnitude").alias("sum_magnitude"),
        F.avg("magnitude").alias("avg_magnitude"),
        F.max("magnitude").alias("max_magnitude"),
        F.sum("depth_km").alias("depth_sum"),
        F.avg("depth_km").alias("avg_depth"),
        F.min("depth_km").alias("depth_min"),
        F.max("depth_km").alias("depth_max"),
        F.max(F.struct(
            F.col("magnitude"), 
            F.col("id"), 
            F.col("place")
        )).alias("max_event_struct")
    )

df_stats = df_agg.select(
    F.col("cluster_id"),
    F.col("dt"),
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
    F.col("max_event_struct.place").alias("max_mag_event_place")
)

# Join Stats + Metadata (Denormalization)
df_final_output = df_stats.join(df_metadata, on="cluster_id", how="inner").select(
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
    F.col("max_mag_event_id"),
    F.col("max_mag_event_place"),
    F.col("dt"),
    F.col("center_lat").alias("centroid_lat"),
    F.col("center_lon").alias("centroid_lon"),
    F.col("max_r_km")
)

# 8. Write to Hive (Flat Table)
output_path = "/user/testuser/earthquake_data/cluster_daily_analytics"
print(f"Writing denormalized analytics to {output_path}...")

df_final_output.coalesce(1).write.mode("overwrite").parquet(output_path)

print(f"Process Complete. Total Time: {time.time() - total_start_time:.2f}s")
spark.stop()