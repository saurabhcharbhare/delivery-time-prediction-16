"""Creates a SYNTHETIC delivery dataset (replace with a real Kaggle dataset later)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 5000
df = pd.DataFrame({
    "distance_km": rng.uniform(0.5, 15, n).round(1),
    "prep_time_min": rng.integers(5, 35, n),
    "traffic": rng.choice([0, 1, 2], n, p=[0.35, 0.40, 0.25]),   # 0 low, 1 medium, 2 high
    "weather": rng.choice([0, 1, 2], n, p=[0.60, 0.25, 0.15]),   # 0 clear, 1 rain, 2 fog
    "peak_hour": rng.choice([0, 1], n, p=[0.65, 0.35]),
    "courier_exp_years": rng.integers(0, 8, n),
})
traffic_f = df.traffic.map({0: 1.0, 1: 1.3, 2: 1.7})
weather_f = df.weather.map({0: 1.0, 1: 1.25, 2: 1.15})
t = df.prep_time_min + df.distance_km * 3.2 * traffic_f * weather_f + 4 * df.peak_hour \
    - 0.8 * np.minimum(df.courier_exp_years, 5) + rng.normal(0, 3, n)
df["delivery_time_min"] = t.round(1).clip(lower=8)
df.to_csv("delivery_data.csv", index=False)
print("Saved delivery_data.csv", df.shape)
