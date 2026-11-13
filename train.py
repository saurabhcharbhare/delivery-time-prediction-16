"""Train and compare 3 models, save best model, metrics and charts.
Usage:  python train.py                 (uses delivery_data.csv)
        python train.py my_real.csv     (real dataset: needs the same 7 column names, see README)"""
import os, sys, json, joblib, numpy as np, pandas as pd
os.makedirs("static", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

path = sys.argv[1] if len(sys.argv) > 1 else "delivery_data.csv"
df = pd.read_csv(path)
TARGET = "delivery_time_min"
X, y = df.drop(columns=TARGET), df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
          "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)}
res = {}
for name, m in models.items():
    cv = -cross_val_score(m, Xtr, ytr, cv=5, scoring="neg_mean_absolute_error").mean()
    m.fit(Xtr, ytr); p = m.predict(Xte)
    res[name] = dict(MAE=round(mean_absolute_error(yte, p), 2),
                     RMSE=round(float(np.sqrt(mean_squared_error(yte, p))), 2),
                     R2=round(r2_score(yte, p), 3), CV_MAE=round(float(cv), 2))
    print(name, res[name])
best = min(res, key=lambda k: res[k]["MAE"])
bm = models[best]; print("Best model:", best)
joblib.dump(dict(model=bm, columns=list(X.columns)), "model.joblib")
json.dump(dict(best=best, results=res, rows=len(df), features=list(X.columns)), open("metrics.json", "w"), indent=1)

# Chart 1: model comparison
plt.figure(figsize=(6, 3.4)); names = list(res)
plt.bar(names, [res[n]["MAE"] for n in names], color=["#1f2a44", "#f26b21", "#5b8def"])
plt.ylabel("MAE (minutes, lower is better)"); plt.title("Model comparison"); plt.tight_layout()
plt.savefig("static/model_comparison.png", dpi=120); plt.close()
# Chart 2: actual vs predicted
p = bm.predict(Xte); plt.figure(figsize=(4.6, 4.2))
plt.scatter(yte, p, s=6, alpha=.4, color="#1f2a44"); lim = [yte.min(), yte.max()]
plt.plot(lim, lim, color="#f26b21"); plt.xlabel("Actual (min)"); plt.ylabel("Predicted (min)")
plt.title("Actual vs predicted (" + best + ")"); plt.tight_layout(); plt.savefig("static/actual_vs_predicted.png", dpi=120); plt.close()
# Chart 3: feature importance (works for every model)
imp = permutation_importance(bm, Xte, yte, n_repeats=5, random_state=42).importances_mean
order = np.argsort(imp); plt.figure(figsize=(6, 3.4))
plt.barh(np.array(X.columns)[order], imp[order], color="#f26b21"); plt.title("Feature importance")
plt.tight_layout(); plt.savefig("static/feature_importance.png", dpi=120); plt.close()
print("Saved model.joblib, metrics.json and 3 charts in static/")
