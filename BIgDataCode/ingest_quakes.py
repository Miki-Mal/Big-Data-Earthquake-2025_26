import math
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, DoubleType, IntegerType
from pyspark.sql.functions import col, from_json, to_date, when, to_timestamp, broadcast, udf
from pyspark.broadcast import Broadcast

# 1. Initialize Spark with Hive Support
spark = SparkSession.builder \
    .appName("EarthquakeRegionClassifier") \
    .config("spark.cassandra.connection.host", "127.0.0.1") \
    .config("spark.cassandra.connection.port", "9042") \
    .enableHiveSupport() \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ==========================================
# 2. LOAD STATIC DATA
# ==========================================

# 2a. Load Region Definitions (CSV)
region_schema = StructType([
    StructField("region_id", IntegerType(), True),
    StructField("region", StringType(), True),
    StructField("sub_region", StringType(), True),
    StructField("lat_min", DoubleType(), True),
    StructField("lat_max", DoubleType(), True),
    StructField("lon_min", DoubleType(), True),
    StructField("lon_max", DoubleType(), True)
])

static_regions_df = spark.read \
    .schema(region_schema) \
    .option("header", "true") \
    .csv("/user/vagrant/static_data/region_definitions.csv")

# 2b. Load Cluster Metadata from Hive
clusters_df = spark.table("dim_cluster_metadata")

# 2c. Broadcast the Hive Table
# Fetch to Driver and Broadcast to all Executors
clusters_list = clusters_df.collect()
bc_clusters = spark.sparkContext.broadcast(clusters_list)



# ==========================================
# 3. DEFINE ROBUST UDF FOR CLUSTER MATCHING
# ==========================================

def find_cluster_logic(lat, lon, depth, clusters_rows):
    """
    Calculates 3D Cartesian distance and finds the first matching cluster.
    Includes STRICT handling for None/Null values to prevent streaming crashes.
    """
    # 1. Validate Stream Input
    if lat is None or lon is None:
        return None
    
    try:
        # Earth Radius in KM
        R_EARTH_KM = 6371.0
        
        # Ensure Earthquake inputs are float (handle string/decimal types)
        lat = float(lat)
        lon = float(lon)
        # Handle null depth (surface)
        current_depth = float(depth) if depth is not None else 0.0
        
        # 2. Convert Earthquake to Cartesian (Unit Sphere)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        r_factor = (R_EARTH_KM - current_depth) / R_EARTH_KM
        
        q_x = r_factor * math.cos(lat_rad) * math.cos(lon_rad)
        q_y = r_factor * math.cos(lat_rad) * math.sin(lon_rad)
        q_z = r_factor * math.sin(lat_rad)
        
        # 3. Iterate through Broadcasted Clusters
        for row in clusters_rows:
            # --- CRITICAL FIX: NULL HANDLING ---
            # We wrap the data extraction in a try/except.
            # If Hive data is None, float() raises TypeError/ValueError.
            # We catch it and 'continue' (skip this bad cluster row).
            try:
                c_x = float(row.center_x)
                c_y = float(row.center_y)
                c_z = float(row.center_z)
                max_r = float(row.max_r_km)
            except (TypeError, ValueError, AttributeError):
                continue # Skip this dirty cluster row
            
            # 4. Calculate Distance (Now guaranteed to be safe floats)
            dist_sq = (q_x - c_x)**2 + (q_y - c_y)**2 + (q_z - c_z)**2
            
            # Normalize Threshold
            threshold_sq = (max_r / R_EARTH_KM)**2
            
            # Check condition
            if dist_sq < threshold_sq:
                # Return standard python int
                return int(row.cluster_id)
                
    except Exception:
        # If any other math error occurs, return None rather than crashing the stream
        return None
            
    return None # No matching cluster found

# Wrap logic to pass the broadcast variable
def find_cluster_wrapper(lat, lon, depth):
    return find_cluster_logic(lat, lon, depth, bc_clusters.value)

# Register UDF
find_cluster_udf = udf(find_cluster_wrapper, IntegerType())

# ==========================================
# 4. STREAMING LOGIC
# ==========================================

input_schema = StructType([
    StructField("id", StringType(), True),
    StructField("time_utc", StringType(), True), 
    StructField("magnitude", FloatType(), True),
    StructField("mag_type", StringType(), True),
    StructField("place", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("depth_km", DoubleType(), True),
    StructField("source", StringType(), True)
])

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "unified_quakes") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse JSON
json_df = kafka_df.select(
    from_json(col("value").cast("string"), input_schema).alias("data")
).select("data.*")

# Clean Data
cleaned_stream_df = json_df.withColumn("time_parsed", 
    when(col("time_utc").rlike("^\d+$"), 
         (col("time_utc").cast("double") / 1000).cast("timestamp")) 
    .otherwise(to_timestamp(col("time_utc")))
) \
.withColumn("day", to_date(col("time_parsed"))) \
.filter(col("day").isNotNull() & col("id").isNotNull()) \
.select(
    col("day"), 
    col("id"), 
    col("time_parsed").alias("time_utc"),
    "magnitude", "mag_type", "place", "latitude", "longitude", "depth_km", "source"
)

# ==========================================
# 5. ENRICHMENT & WRITE
# ==========================================

# Join Regions
enriched_df = cleaned_stream_df.join(
    broadcast(static_regions_df),
    (cleaned_stream_df.latitude >= static_regions_df.lat_min) &
    (cleaned_stream_df.latitude <= static_regions_df.lat_max) &
    (cleaned_stream_df.longitude >= static_regions_df.lon_min) &
    (cleaned_stream_df.longitude <= static_regions_df.lon_max),
    "left"
)

# Apply Cluster UDF
final_df = enriched_df.withColumn(
    "cluster_id", 
    find_cluster_udf(col("latitude"), col("longitude"), col("depth_km"))
).select(
    "day", "id", "time_utc", "magnitude", "mag_type", 
    "place", "latitude", "longitude", "depth_km", "source",
    "region_id", "region", "sub_region",
    "cluster_id"
)

# Write to Cassandra
query = final_df.writeStream \
    .trigger(processingTime='5 seconds') \
    .format("org.apache.spark.sql.cassandra") \
    .option("keyspace", "earthquakes") \
    .option("table", "events") \
    .option("checkpointLocation", "/tmp/checkpoints/unified_quakes_complete") \
    .option("spark.cassandra.output.consistency.level", "ONE") \
    .outputMode("append") \
    .start()

print("-------------------------------------------------------")
print(" Spark Streaming is running...")
print(" Status: Robust Null-Handling Enabled")
print(" Writing to Cassandra table: earthquakes.events")
print("-------------------------------------------------------")

query.awaitTermination()