# predictive_heatmap.py
"""
سكريبت عملي: يولّد بيانات تجريبية، يدرب موديل، ويرسم خريطة حرارية تنبؤية باستخدام folium.
شغله: python predictive_heatmap.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import folium
from folium.plugins import HeatMap
from math import radians, cos, sin, asin, sqrt

# ---------- 1) توليد بيانات تجريبية ----------
np.random.seed(0)

# نحدد منطقة (مثال: جزء بسيط من القاهرة) بالإحداثيات (lat, lon) تقريبيًا
lat_min, lat_max = 30.00, 30.08
lon_min, lon_max = 31.20, 31.30

# ننشئ 5 محطات (cell towers) عشوائية داخل المنطقة
n_towers = 5
towers = pd.DataFrame({
    'tower_id': range(n_towers),
    'lat': np.random.uniform(lat_min, lat_max, n_towers),
    'lon': np.random.uniform(lon_min, lon_max, n_towers),
    'power_dbm': np.random.uniform(40, 50, n_towers)  # قوة الإرسال (مقاس نموذجي)
})

# ننشئ 1000 قياس نقطة موزعة في المنطقة
n_measurements = 1000
measurements = pd.DataFrame({
    'lat': np.random.uniform(lat_min, lat_max, n_measurements),
    'lon': np.random.uniform(lon_min, lon_max, n_measurements),
    'timestamp': pd.Timestamp('2025-10-15') + pd.to_timedelta(np.random.randint(0, 86400, n_measurements), unit='s')
})

# دالة لحساب المسافة بين نقطتين (هارفارس) بالكيلومتر
def haversine(lat1, lon1, lat2, lon2):
    # تحويل إلى راديان
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    R = 6371  # Earth radius km
    return R * c

# نحسب أقرب برج لكل قياس ونولّد قيمة إشارة dBm باستخدام نموذج تبعثر بسيط
def simulate_signal(row):
    # احسب المسافات لكل البرج
    dists = towers.apply(lambda t: haversine(row.lat, row.lon, t.lat, t.lon), axis=1)
    nearest_idx = dists.idxmin()
    dist_km = dists[nearest_idx]
    tower_power = towers.loc[nearest_idx, 'power_dbm']
    # نموذج خسارة بسيط: RSSI = power - 20*log10(distance_km*1000) - 30 (ثابت للبيئة) + noise
    if dist_km < 0.001:
        dist_km = 0.001
    path_loss = 20 * np.log10(dist_km * 1000) + 30
    rssi = tower_power - path_loss + np.random.normal(0, 2)  # ضوضاء عشوائية sigma=2 dB
    return pd.Series({'rssi_dbm': rssi, 'nearest_tower': nearest_idx, 'dist_km': dist_km})

sim = measurements.apply(simulate_signal, axis=1)
measurements = pd.concat([measurements, sim], axis=1)

# تحويل dBm إلى مقياس أكثر ودودًا للتلوين (مثلاً نسبة 0-1)
# عادة dBm يتراوح: -120 (سيئ) إلى -50 (قوي). نعمل scaling.
def rssi_to_score(rssi, low=-120, high=-50):
    score = (rssi - low) / (high - low)
    score = np.clip(score, 0, 1)
    return score

measurements['score'] = measurements['rssi_dbm'].apply(rssi_to_score)

# ---------- 2) تجهيز الميزات (features) ----------
# نستخدم: lat, lon, hour_of_day, dist_to_nearest_tower
measurements['hour'] = measurements['timestamp'].dt.hour

X = measurements[['lat', 'lon', 'dist_km', 'hour']]
y = measurements['score']

# تقسيم تدريب/اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---------- 3) تدريب موديل بسيط ----------
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# تقييم
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f}, R2: {r2:.4f}")

# ---------- 4) عمل شبكة نقاط (grid) للتنبؤ عليها ----------
grid_lat = np.linspace(lat_min, lat_max, 80)  # كلما زودت الأرقام، زادت الدقة
grid_lon = np.linspace(lon_min, lon_max, 80)
grid = []
for lat in grid_lat:
    for lon in grid_lon:
        # نحسب المسافة لأقرب برج
        dists = towers.apply(lambda t: haversine(lat, lon, t.lat, t.lon), axis=1)
        dist_km = dists.min()
        # نختار ساعة ثابتة (مثال)؛ لو عندك بيانات زمنية استخدم متغيرات مختلفة
        hour = 12
        grid.append([lat, lon, dist_km, hour])

grid_df = pd.DataFrame(grid, columns=['lat', 'lon', 'dist_km', 'hour'])
grid_df['pred_score'] = model.predict(grid_df[['lat','lon','dist_km','hour']])
# نحول score لدرجة لونية (0-1) جاهزة للHeatMap
heat_data = grid_df[['lat','lon','pred_score']].values.tolist()

# ---------- 5) رسم الخريطة الحرارية باستخدام folium ----------
# نحدد مركز الخريطة
center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

# نضيف نقاط المحطات كدوائر
for _, t in towers.iterrows():
    folium.CircleMarker(
        location=[t.lat, t.lon],
        radius=7,
        popup=f"Tower {int(t.tower_id)} Power {t.power_dbm:.1f} dBm",
        fill=True
    ).add_to(m)

# نضيف HeatMap (weight = pred_score)
HeatMap([[r[0], r[1], r[2]] for r in heat_data], radius=12, blur=15, max_zoom=13).add_to(m)

# حفظ الخريطة كـ HTML
out_file = "predictive_heatmap.html"
m.save(out_file)
print(f"Saved predictive heatmap to {out_file}")
