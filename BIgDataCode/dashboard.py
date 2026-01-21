import streamlit as st
import pandas as pd
import plotly.express as px
from pyhive import hive
from cassandra.cluster import Cluster
from cassandra.util import Date as CassDate, Time as CassTime
import time as sys_time
from datetime import datetime, date, time as dt_time
import math
from decimal import Decimal
import warnings

# --- 🔇 Silence Warnings ---
warnings.filterwarnings('ignore')

# --- ⚙️ Configuration ---
# Cassandra
CASSANDRA_HOSTS = ['127.0.0.1']
CASSANDRA_PORT = 9042
KEYSPACE = 'earthquakes'

# Hive
HIVE_HOST = 'localhost'
HIVE_PORT = 10000 
HIVE_USER = 'testuser'

st.set_page_config(page_title="Seismic Master Dashboard", layout="wide", page_icon="🌋")

# ==========================================
# 🔌 CONNECTION FUNCTIONS
# ==========================================

@st.cache_resource
def get_cassandra_session():
    """Connects to Cassandra."""
    try:
        cluster = Cluster(contact_points=CASSANDRA_HOSTS, port=CASSANDRA_PORT)
        session = cluster.connect()
        return session
    except Exception as e:
        st.error(f"❌ Cassandra Connection Failed: {e}")
        return None

@st.cache_resource
def get_hive_conn():
    """Connects to Hive (Local Mode)."""
    try:
        conn = hive.Connection(host=HIVE_HOST, port=HIVE_PORT, username=HIVE_USER, auth='NONE')
        cursor = conn.cursor()
        cursor.execute("SET mapreduce.framework.name=local") 
        cursor.execute("SET hive.exec.mode.local.auto=true")
        cursor.close()
        return conn
    except Exception as e:
        st.error(f"❌ Hive Connection Failed: {e}")
        return None

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def normalize_cassandra_row(row):
    """Converts Cassandra types to Python types."""
    data = {}
    for field in row._fields:
        value = getattr(row, field)
        if isinstance(value, (CassDate, date)):
            value = str(value)
        elif isinstance(value, (CassTime, dt_time)):
            value = str(value)
        elif isinstance(value, Decimal):
            value = float(value)
        data[field] = value
    return data

def rerun():
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()

# ==========================================
# 📥 DATA LOADERS
# ==========================================

def fetch_live_cassandra_data(session, table_name, limit=1000):
    try:
        # 1. Fetch the large buffer
        query = f"SELECT * FROM {KEYSPACE}.{table_name} LIMIT {limit}"
        rows = session.execute(query)
        clean_data = [normalize_cassandra_row(r) for r in rows]
        df = pd.DataFrame(clean_data)
        
        # 2. Sort by time (Newest first)
        if not df.empty:
            if 'time_utc' in df.columns:
                df['time_utc'] = pd.to_datetime(df['time_utc'])
                df = df.sort_values(by='time_utc', ascending=False)
            else:
                time_col = next((col for col in df.columns if 'time' in col.lower()), None)
                if time_col:
                    df[time_col] = pd.to_datetime(df[time_col])
                    df = df.sort_values(by=time_col, ascending=False)
        
        # 3. Capture Stats & Slice
        total_count = len(df)       # The "Buffer" size (e.g., 1000)
        df_display = df.head(10)    # The "View" (Top 10 only)

        # 4. Return BOTH
        return df_display, total_count

    except Exception as e:
        st.error(f"Cassandra Query Error: {e}")
        return pd.DataFrame(), 0

@st.cache_data(ttl=3600)
def load_hive_dates():
    conn = get_hive_conn()
    if not conn: return []
    try:
        df = pd.read_sql("SELECT dt FROM region_daily_analytics LIMIT 10000", conn)
        if not df.empty:
            return sorted(df['dt'].astype(str).unique(), reverse=True)
    except:
        return []
    return []

@st.cache_data(ttl=600)
def load_hive_dashboard_data(selected_date):
    conn = get_hive_conn()
    if not conn: return pd.DataFrame(), pd.DataFrame()

    # 1. Clusters Data
    q_map = f"""
        SELECT cluster_id, total_events, max_magnitude, avg_depth, 
               centroid_lat, centroid_lon, max_r_km, max_mag_event_place
        FROM cluster_daily_analytics 
        WHERE dt = '{selected_date}'
    """
    
    # 2. Regions Data
    q_stats = f"""
        SELECT region, sub_region, total_events, max_magnitude, avg_magnitude 
        FROM region_daily_analytics 
        WHERE dt = '{selected_date}'
    """
    
    try:
        df_map = pd.read_sql(q_map, conn)
        df_stats = pd.read_sql(q_stats, conn)

        # Cleaning
        if not df_map.empty:
            df_map['centroid_lat'] = pd.to_numeric(df_map['centroid_lat'], errors='coerce')
            df_map['centroid_lon'] = pd.to_numeric(df_map['centroid_lon'], errors='coerce')
            df_map['total_events'] = pd.to_numeric(df_map['total_events'], errors='coerce').fillna(10)
            df_map = df_map.dropna(subset=['centroid_lat', 'centroid_lon'])

        if not df_stats.empty:
            df_stats['total_events'] = pd.to_numeric(df_stats['total_events'], errors='coerce').fillna(0)
            df_stats['max_magnitude'] = pd.to_numeric(df_stats['max_magnitude'], errors='coerce').fillna(0)

        return df_map, df_stats
    except Exception as e:
        st.error(f"Hive Data Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ==========================================
# 🖥️ MAIN UI LAYOUT
# ==========================================

st.title("🌋 Seismic Master Dashboard")

# Custom CSS for metrics
st.markdown("""
    <style>
        div[data-testid="metric-container"] {
            background-color: #1E1E1E; 
            border: 1px solid #4A4A4A; 
            padding: 15px;
            border-radius: 8px;
            min-height: 100px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        div[data-testid="metric-container"] label { color: #B0B0B0 !important; }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------
# SECTION 1: REAL-TIME (CASSANDRA)
# ------------------------------------------
st.subheader("📡 Real-Time Live Feed")

cass_session = get_cassandra_session()
if cass_session:
    try:
        tables = cass_session.execute(f"SELECT table_name FROM system_schema.tables WHERE keyspace_name='{KEYSPACE}'")
        table_list = [row.table_name for row in tables]
    except:
        table_list = []

    if table_list:
        target_table = table_list[0]
        
        # Unpack the TWO return values: the Dataframe and the Count
        df_live, buffer_count = fetch_live_cassandra_data(cass_session, target_table, limit=1000)

        if not df_live.empty:
            now_date = now_date = datetime.now().strftime("%Y-%m-%d")
            
            # Calculate Max Magnitude (Based on the top 10 displayed rows)
            mag_col = next((col for col in df_live.columns if 'mag' in col.lower() and 'type' not in col.lower()), None)
            if mag_col:
                df_live[mag_col] = pd.to_numeric(df_live[mag_col], errors='coerce')
                max_val = round(df_live[mag_col].max(), 1)
                max_mag_live = max_val if not (pd.isna(max_val) or math.isnan(max_val)) else 0.0
            else:
                max_mag_live = 0.0

            # Display Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("📅 Today's Date", f"{now_date}")
            m2.metric("📊 Live Events (Buffer)", f"{buffer_count}")  # Uses the total count returned
            m3.metric("💥 Max Magnitude (Top 10)", f"{max_mag_live} M")
            
            # Display Table (Already sliced to 10 by the function)
            st.dataframe(df_live, use_container_width=True)
        
        else:
            st.warning("Waiting for live data stream...")
            
    else:
        st.error("No Cassandra tables found.")
else:
    st.error("Cassandra connection failed.")

# ------------------------------------------
# SECTION 2: HISTORICAL (HIVE)
# ------------------------------------------
st.subheader("📜 Historical Analysis")

# Load Dates
hive_dates = load_hive_dates()

if hive_dates:
    # Row 1: Filters & Metrics
    h1, h2, h3 = st.columns(3)
    
    with h1:
        # 1. Date Filter
        selected_date = st.selectbox("📅 Select Historical Date", hive_dates)
        
        # Load Data FIRST so we can find available clusters
        df_clusters, df_regions = load_hive_dashboard_data(selected_date)
        
        # 2. Cluster Filter (Dynamic)
        selected_clusters = []
        if not df_clusters.empty:
            all_clusters = sorted(df_clusters['cluster_id'].unique())
            # Multiselect allows picking one or many
            selected_clusters = st.multiselect("🔍 Filter Cluster ID", all_clusters, placeholder="Show All Clusters")

    # Apply Filter to Visuals (Map & Table)
    if not df_clusters.empty:
        if selected_clusters:
            df_clusters_viz = df_clusters[df_clusters['cluster_id'].isin(selected_clusters)]
        else:
            df_clusters_viz = df_clusters
    else:
        df_clusters_viz = pd.DataFrame()

    # Display Metrics (We keep these Global for the Date context)
    if not df_clusters.empty and not df_regions.empty:
        total_events_hist = df_regions['total_events'].sum()
        max_mag_hist = df_clusters['max_magnitude'].max()
        
        h2.metric("🌍 Global Events (Hist)", f"{total_events_hist:,}")
        h3.metric("💥 Max Magnitude (Hist)", f"{max_mag_hist} M")
        
        st.markdown("") # Spacing

        # --------------------
        # Visualizations
        # --------------------
        c_map, c_stats = st.columns([2, 1])

        with c_map:
            st.markdown("#### 📍 Cluster Activity Map")
            # Uses the FILTERED dataframe
            if not df_clusters_viz.empty:
                map_data = df_clusters_viz.rename(columns={'centroid_lat': 'latitude', 'centroid_lon': 'longitude'})
                st.map(map_data, latitude='latitude', longitude='longitude', size='total_events', zoom=1)
            else:
                st.info("No clusters match the selected filter.")

        with c_stats:
            st.markdown("#### 📊 Events by Region")
            # Uses the GLOBAL dataframe (as requested, only impact Map & Cluster Table)
            df_regions_vol = df_regions.sort_values("total_events", ascending=True).tail(15)
            fig_bar = px.bar(
                df_regions_vol, x="total_events", y="region", orientation='h',
                color="max_magnitude", hover_data=["sub_region", "avg_magnitude"]
            )
            fig_bar.update_layout(height=400, yaxis_title=None, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # --------------------
        # Tables
        # --------------------
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("#### 🏆 Top Regions")
            # --- FIX 2: Added 'sub_region' to the column list below ---
            st.dataframe(
                df_regions.sort_values("total_events", ascending=False).head(10)[['region', 'sub_region', 'total_events', 'max_magnitude']], 
                hide_index=True, 
                use_container_width=True
            )
        with t2:
            st.markdown("#### 🔥 Top Clusters")
            # Uses the FILTERED dataframe
            if not df_clusters_viz.empty:
                st.dataframe(df_clusters_viz.sort_values("max_magnitude", ascending=False).head(10)[['cluster_id', 'max_magnitude', 'total_events', 'max_mag_event_place']], hide_index=True, use_container_width=True)
            else:
                st.info("No data for selected cluster(s).")
            
    else:
        st.warning(f"No historical data found for {selected_date}")
else:
    st.info("No historical dates available in Hive.")

# ------------------------------------------
# 🔄 AUTO-REFRESH LOGIC
# ------------------------------------------
# This keeps the Live section updating.
sys_time.sleep(5)
rerun()