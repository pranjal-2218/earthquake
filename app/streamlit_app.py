import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from src.data.fetch_data import USGSDataFetcher
from src.utils.config import Config

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("DEBUG PATH:", sys.path)

# Page Configuration
st.set_page_config(
    page_title="SeismoCast | Earthquake Predictive Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphism and Card Styling
st.markdown("""
<style>
    /* Card design */
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #888888;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-value-accent {
        color: #ff4b4b !important;
    }
    .metric-value-safe {
        color: #00e676 !important;
    }
    /* Section headers */
    h2, h3 {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to check API health
def check_api_health():
    try:
        response = requests.get(f"{Config.API_URL}/")
        return response.status_code == 200
    except:
        return False

# Header Section
st.title("🌍 SeismoCast")
st.caption("Advanced Real-time Earthquake Risk Classification and Magnitude Prediction System")

# Sidebar Configuration
st.sidebar.header("🕹️ Control Center")
st.sidebar.markdown("Configure location inputs to predict seismic risk.")

latitude = st.sidebar.slider("Latitude", min_value=-90.0, max_value=90.0, value=20.0, step=0.1)
longitude = st.sidebar.slider("Longitude", min_value=-180.0, max_value=180.0, value=78.0, step=0.1)
depth = st.sidebar.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=10.0, step=1.0)

# Collapsible advanced physical parameters panel
with st.sidebar.expander("🛠️ Advanced Parameters (Physical Signature)"):
    use_custom = st.checkbox("Provide custom physical features", value=False, help="Provide known properties of the seismic event to improve accuracy. Otherwise, training medians will be imputed.")
    if use_custom:
        nst = st.number_input("Number of Reporting Stations (nst)", min_value=0, max_value=1000, value=19)
        tsunami = st.selectbox("Tsunami Warning Generated", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        rms = st.number_input("Travel Time Residual (rms)", min_value=0.0, max_value=5.0, value=0.18, step=0.01)
        gap = st.number_input("Azimuthal Gap (gap)", min_value=0.0, max_value=360.0, value=90.0, step=1.0)
        dmin = st.number_input("Min Distance to Station (dmin)", min_value=0.0, max_value=50.0, value=0.07, step=0.01)
    else:
        nst = None
        tsunami = None
        rms = None
        gap = None
        dmin = None

# Check backend status
api_online = check_api_health()
if api_online:
    st.sidebar.success("● API Service: ONLINE")
else:
    st.sidebar.error("○ API Service: OFFLINE")
    st.sidebar.warning("Please start the backend API server (`python -m uvicorn src.api.main:app`) to enable predictions.")

# Fetch USGS data for mapping and analytics
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_cached_data():
    try:
        fetcher = USGSDataFetcher()
        return fetcher.fetch()
    except Exception as e:
        st.error(f"Failed to fetch historical earthquake data: {str(e)}")
        return pd.DataFrame()

df_historical = get_cached_data()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Real-time Prediction",
    "📈 Spatial Analytics & Trends",
    "📊 Model Performance"
])

# ----------------- TAB 1: Real-time Prediction -----------------
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Compute Seismic Risk")
        st.markdown(
            "Predicting if this geographical coordinate is prone to high-magnitude earthquakes (>5.0 magnitude) "
            "and calculating estimated magnitude."
        )
        
        predict_btn = st.button("Run Prediction", type="primary", disabled=not api_online)
        
        if predict_btn and api_online:
            with st.spinner("Analyzing seismic patterns..."):
                try:
                    payload = {
                        "latitude": latitude,
                        "longitude": longitude,
                        "depth": depth,
                        "nst": nst,
                        "tsunami": tsunami,
                        "rms": rms,
                        "gap": gap,
                        "dmin": dmin
                    }
                    response = requests.post(f"{Config.API_URL}/predict", json=payload)
                    
                    if response.status_code == 200:
                        res = response.json()
                        prone = res["earthquake_prone"]
                        magnitude = res["predicted_magnitude"]
                        
                        # Render Risk Card
                        risk_class = "metric-value-accent" if prone == 1 else "metric-value-safe"
                        risk_text = "High Risk (> 5.0 Mag)" if prone == 1 else "Low/Safe Risk"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Seismic Classification</div>
                            <div class="metric-value {risk_class}">{risk_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Render Magnitude Card
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Predicted Magnitude</div>
                            <div class="metric-value">{magnitude:.2f} <span style="font-size:1rem;color:#888;">Richter Scale</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {str(e)}")
        elif not api_online:
            st.info("API is offline. Predictions are disabled.")

    with col2:
        st.subheader("Geospatial Visualization")
        # Base map centered at selection
        m = folium.Map(location=[latitude, longitude], zoom_start=4, control_scale=True)
        
        # User input marker
        folium.Marker(
            [latitude, longitude], 
            tooltip="Prediction Target Location",
            popup=f"Target Coordinates:<br>Lat: {latitude:.4f}<br>Lon: {longitude:.4f}<br>Depth: {depth} km",
            icon=folium.Icon(color="red", icon="crosshair", prefix="fa")
        ).add_to(m)

        # Plot recent nearby earthquakes (if data loaded)
        if not df_historical.empty:
            # Filter nearby earthquakes within ~10 degrees distance
            nearby = df_historical[
                (df_historical["latitude"] - latitude).abs() < 10
            ].head(50)  # limit to 50 for performance
            
            for _, row in nearby.iterrows():
                mag = row["magnitude"]
                # Color code based on magnitude
                color = "#ffeb3b" if mag < 4 else ("#ff9800" if mag < 5 else "#f44336")
                
                folium.Circle(
                    location=[row["latitude"], row["longitude"]],
                    radius=float(mag * 30000),  # Scale radius based on magnitude
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.3,
                    tooltip=f"Magnitude: {mag} | Depth: {row['depth']}km",
                ).add_to(m)
        
        # Render the map
        st_folium(m, height=450, use_container_width=True)

# ----------------- TAB 2: Spatial Analytics & Trends -----------------
with tab2:
    if df_historical.empty:
        st.warning("No historical data available for analytics.")
    else:
        st.subheader("Global Seismic Insights (USGS Live Month Summary)")
        
        # Top-level Stats
        total_eq = len(df_historical)
        avg_mag = df_historical["magnitude"].mean()
        max_mag = df_historical["magnitude"].max()
        high_risk_eq = len(df_historical[df_historical["magnitude"] > 5.0])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Earthquakes (30 Days)", f"{total_eq:,}")
        c2.metric("Average Magnitude", f"{avg_mag:.2f}")
        c3.metric("Max Recorded Magnitude", f"{max_mag:.1f}")
        c4.metric("High Magnitude (Mag > 5)", f"{high_risk_eq}")
        
        st.markdown("---")
        
        col_charts_1, col_charts_2 = st.columns(2)
        
        with col_charts_1:
            st.write("📊 **Distribution of Earthquake Magnitudes**")
            mag_counts, bin_edges = np.histogram(df_historical["magnitude"], bins=15)
            chart_df = pd.DataFrame({
                "Magnitude Range": [f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)],
                "Count": mag_counts
            })
            st.bar_chart(chart_df.set_index("Magnitude Range"), color="#ff9800")
            
        with col_charts_2:
            st.write("🌋 **Depth vs. Magnitude Correlation**")
            chart_data = df_historical[["magnitude", "depth"]].copy()
            # Binning depth for scatter visualization in Streamlit's line/area/bar or just standard scatter
            st.scatter_chart(
                chart_data,
                x="magnitude",
                y="depth",
                color="#ff4b4b"
            )

# ----------------- TAB 3: Model Performance -----------------
with tab3:
    st.subheader("Production Model Metrics & Verification")
    st.markdown(
        "These metrics are generated automatically during model training and validation. "
        "They represent out-of-sample performance on the holdout test dataset (20% split)."
    )
    
    if api_online:
        try:
            res_info = requests.get(f"{Config.API_URL}/model-info").json()
            metrics = res_info["metrics"]
            metadata = res_info["model_metadata"]
            
            st.markdown(f"**Model Type**: `{metadata['algorithm']}`")
            st.markdown(f"**Trained Features**: `{', '.join(metadata['features'])}`")
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("### 🎯 Binary Classifier Metrics")
                st.markdown("*(Risk Threshold: Magnitude > 5.0)*")
                
                cl1, cl2 = st.columns(2)
                cl1.metric("Accuracy", f"{metrics['classifier']['accuracy']:.2%}")
                cl2.metric("F1-Score", f"{metrics['classifier']['f1_score']:.3f}")
                
                cl3, cl4 = st.columns(2)
                cl3.metric("Precision", f"{metrics['classifier']['precision']:.3f}")
                cl4.metric("Recall", f"{metrics['classifier']['recall']:.3f}")
                
            with col_m2:
                st.markdown("### 📊 Continuous Regressor Metrics")
                st.markdown("*(Predicting Richter Scale Magnitude)*")
                
                rl1, rl2 = st.columns(2)
                rl1.metric("Mean Absolute Error (MAE)", f"{metrics['regressor']['mae']:.3f}")
                rl2.metric("Root Mean Sq. Error (RMSE)", f"{metrics['regressor']['rmse']:.3f}")
                
                rl3 = st.columns(1)
                st.metric("R² Score (Explained Variance)", f"{metrics['regressor']['r2_score']:.2%}")
                
        except Exception as e:
            st.warning("Failed to parse model information from the API. Has the model been trained and saved?")
            st.exception(e)
    else:
        st.warning(
            "API Service is offline. Could not retrieve real-time model metrics. "
            "Start the API server to query model metrics."
        )
