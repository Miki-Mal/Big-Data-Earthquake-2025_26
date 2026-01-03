import streamlit as st
import pandas as pd
from pyhive import hive
from cassandra.cluster import Cluster
from datetime import date
import plotly.express as px


# =========================
# Config
# =========================
HIVE_HOST = "localhost"
HIVE_PORT = 10000
HIVE_USER = "testuser"

CASSANDRA_HOSTS = ["localhost"]
CASSANDRA_KEYSPACE = "earthquakes"

st.set_page_config(page_title="Analytics Dashboard", page_icon="🛢️")
st.title("🛢️ Hive & 🌍 Cassandra Analytics")

# =========================
# Hive
# =========================
@st.cache_resource
def get_hive_connection():
    return hive.Connection(
        host=HIVE_HOST,
        port=HIVE_PORT,
        username=HIVE_USER,
        auth="NONE"
    )

def fetch_hive_data(table, limit=10):
    try:
        cursor = get_hive_connection().cursor()
        cursor.execute(f"SELECT * FROM {table} LIMIT {limit}")
        cols = [c[0] for c in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=cols)
    except Exception as e:
        st.error(e)
        return pd.DataFrame()

# =========================
# Cassandra
# =========================
@st.cache_resource
def get_cassandra_session():
    cluster = Cluster(CASSANDRA_HOSTS)
    return cluster.connect(CASSANDRA_KEYSPACE)

@st.cache_data(ttl=300, show_spinner="Refreshing Cassandra data...")
def load_cassandra_data():
    session = get_cassandra_session()
    query = """
        SELECT day, time_utc, id, cluster_id, depth_km,
               latitude, longitude, mag_type, magnitude,
               place, region, region_id, source, sub_region
        FROM events
    """
    rows = session.execute(query)
    return pd.DataFrame(rows)

# =========================
# Hive UI
# =========================
st.subheader("1️⃣ Cluster Daily Analytics (Hive)")
st.dataframe(fetch_hive_data("cluster_daily_analytics"), use_container_width=True)

st.divider()

st.subheader("2️⃣ Region Daily Analytics (Hive)")
st.dataframe(fetch_hive_data("region_daily_analytics"), use_container_width=True)

# =========================
# Cassandra UI
# =========================
st.divider()
st.subheader("3️⃣ Earthquake Events (Cassandra – auto refresh every 5 min)")

df = load_cassandra_data()

if df.empty:
    st.info("No earthquake data found.")
    st.stop()

# -------------------------
# Filters
# -------------------------
# Ensure correct date type
# Convert Cassandra Date -> python datetime.date
df["day"] = df["day"].apply(lambda d: d.date() if d is not None else None)

st.markdown("### 🔎 Filters")

min_day = min(df["day"])
max_day = max(df["day"])

col1, col2 = st.columns(2)

with col1:
    start_date, end_date = st.date_input(
        "Date range",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day
    )

with col2:
    mag_range = st.slider(
        "Magnitude range",
        min_value=float(df["magnitude"].min()),
        max_value=float(df["magnitude"].max()),
        value=(
            float(df["magnitude"].min()),
            float(df["magnitude"].max())
        ),
        step=0.1
    )

filtered_df = df[
    (df["day"] >= start_date) &
    (df["day"] <= end_date) &
    (df["magnitude"] >= mag_range[0]) &
    (df["magnitude"] <= mag_range[1])
]

# -------------------------
# Prepare data for map
# -------------------------
map_df = filtered_df[
    ["latitude", "longitude", "magnitude", "place", "region"]
].dropna(subset=["latitude", "longitude"])

map_df = map_df.rename(
    columns={"latitude": "lat", "longitude": "lon"}
)

# Ensure numeric types
map_df["lat"] = map_df["lat"].astype(float)
map_df["lon"] = map_df["lon"].astype(float)

st.markdown("### 🌍 Earthquake Map")

if map_df.empty:
    st.info("No events to display on the map.")
else:
    st.map(map_df, zoom=2)


st.caption(f"Showing {len(filtered_df)} events")

st.dataframe(filtered_df, use_container_width=True)
