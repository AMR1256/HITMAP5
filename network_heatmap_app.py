# network_heatmap_app.py
"""
واجهة تفاعلية (Streamlit) لعرض وتحليل الإشارة وعمل خرائط حرارية تنبؤية
بها: 
- Weak Zones Alerts
- Time Analysis
شغلها بالأمر:
python -m streamlit run network_heatmap_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title="AI Predictive Heatmap", layout="wide")

st.title("📡 AI Network Coverage Prediction System")
st.markdown("تحليل بيانات الشبكة ورسم خرائط حرارية تنبؤية مع التنبيهات والتحليل الزمني")

# ------------------------- إعدادات المستخدم -------------------------
st.sidebar.header("⚙️ الإعدادات")

area_lat = st.sidebar.slider("📍 خط العرض (latitude) - المركز", 29.9, 31.0, 30.05)
area_lon = st.sidebar.slider("📍 خط الطول (longitude) - المركز", 30.9, 31.5, 31.25)
area_size = st.sidebar.slider("🗺️ حجم المنطقة (كم)", 0.02, 0.1, 0.05)
n_towers = st.sidebar.slider("📶 عدد الأبراج", 2, 10, 5)
n_measurements = st.sidebar.slider("📊 عدد القياسات", 200, 2000, 800)
selected_hour = st.sidebar.slider("🕒 الساعة للتحليل الزمني", 0, 23, 12)

uploaded = st.sidebar.file_uploader("📂 تحميل ملف بيانات CSV (اختياري)", type="csv")

# ------------------------- وظائف -------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def rssi_to_score(rssi, low=-120, high=-50):
    return np.clip((rssi - low) / (high - low), 0, 1)

# ------------------------- توليد بيانات أو تحميل -------------------------
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.success("✅ تم تحميل البيانات بنجاح!")
    use_real_data = True
else:
    use_real_data = False
    lat_min, lat_max = area_lat - area_size, area_lat + area_size
    lon_min, lon_max = area_lon - area_size, area_lon + area_size

    np.random.seed(0)
    towers = pd.DataFrame({
        "lat": np.random.uniform(lat_min, lat_max, n_towers),
        "lon": np.random.uniform(lon_min, lon_max, n_towers),
        "power_dbm": np.random.uniform(40, 50, n_towers)
    })

    measurements = pd.DataFrame({
        "lat": np.random.uniform(lat_min, lat_max, n_measurements),
        "lon": np.random.uniform(lon_min, lon_max, n_measurements),
        "timestamp": pd.Timestamp("2025-10-15") + pd.to_timedelta(np.random.randint(0, 86400, n_measurements), unit="s")
    })

    def simulate_signal(row):
        dists = towers.apply(lambda t: haversine(row.lat, row.lon, t.lat, t.lon), axis=1)
        nearest_idx = dists.idxmin()
        dist_km = dists[nearest_idx]
        tower_power = towers.loc[nearest_idx, "power_dbm"]
        if dist_km < 0.001:
            dist_km = 0.001
        path_loss = 20 * np.log10(dist_km * 1000) + 30
        rssi = tower_power - path_loss + np.random.normal(0, 2)
        return pd.Series({"rssi_dbm": rssi, "dist_km": dist_km})

    sim = measurements.apply(simulate_signal, axis=1)
    data = pd.concat([measurements, sim], axis=1)
    data["score"] = data["rssi_dbm"].apply(rssi_to_score)
    data["hour"] = data["timestamp"].dt.hour

# ------------------------- تدريب الموديل -------------------------
X = data[["lat", "lon", "dist_km", "hour"]] if "dist_km" in data.columns else data[["lat", "lon"]]
y = data["score"] if "score" in data.columns else data["rssi_dbm"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.sidebar.success("✅ تم تدريب الموديل!")

# ------------------------- إنشاء شبكة تنبؤ -------------------------
grid_lat = np.linspace(data["lat"].min(), data["lat"].max(), 80)
grid_lon = np.linspace(data["lon"].min(), data["lon"].max(), 80)
grid = []
for lat in grid_lat:
    for lon in grid_lon:
        dist = 0.01
        grid.append([lat, lon, dist, selected_hour])

grid_df = pd.DataFrame(grid, columns=["lat", "lon", "dist_km", "hour"])
grid_df["pred_score"] = model.predict(grid_df[["lat", "lon", "dist_km", "hour"]])
grid_df["pred_rssi"] = grid_df["pred_score"] * (-50 + 120) - 120  # تحويل تقريبي

# ------------------------- تنبيهات المناطق الضعيفة -------------------------
weak_zones = grid_df[grid_df["pred_score"] < 0.3]  # أقل من 30% قوة

# ------------------------- رسم الخريطة -------------------------
m = folium.Map(location=[area_lat, area_lon], zoom_start=14, tiles="OpenStreetMap")

# خريطة حرارية
HeatMap(grid_df[["lat", "lon", "pred_score"]].values.tolist(), radius=12, blur=15).add_to(m)

# نقاط الأبراج
if not use_real_data:
    for _, t in towers.iterrows():
        folium.CircleMarker(
            [t.lat, t.lon],
            radius=6,
            color="blue",
            fill=True,
            fill_color="blue",
            popup=f"Tower ({t.lat:.4f}, {t.lon:.4f})"
        ).add_to(m)

# نقاط ضعيفة باللون الأحمر
for _, row in weak_zones.iterrows():
    folium.CircleMarker(
        [row.lat, row.lon],
        radius=4,
        color="red",
        fill=True,
        fill_opacity=0.8,
        popup=f"Weak Zone RSSI≈{row.pred_rssi:.1f} dBm"
    ).add_to(m)

st_folium(m, width=1100, height=600)
st.write("✅ الخريطة تم إنشاؤها بنجاح، لكن لو مش ظاهرة جرب تقلل عدد النقاط أو تحدّث الصفحة.")


# جدول تنبيهات
if not weak_zones.empty:
    st.warning("⚠️ تم اكتشاف مناطق ضعيفة الإشارة:")
    st.dataframe(weak_zones[["lat", "lon", "pred_rssi"]].head(20))
else:
    st.success("✅ لا توجد مناطق ضعيفة في هذا التوقيت")

st.success("🎉 الخريطة التنبؤية جاهزة!")

