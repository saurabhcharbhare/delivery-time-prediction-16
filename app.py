"""Flask backend with REST API and SQLite history.   python app.py  ->  http://127.0.0.1:5000"""
import os, json, sqlite3, joblib, pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder=".")
saved = joblib.load("model.joblib"); MODEL, COLS = saved["model"], saved["columns"]
METRICS = json.load(open("metrics.json"))
ERR = METRICS["results"][METRICS["best"]]["MAE"]
DB = os.environ.get("DB_PATH", "predictions.db")
LIMITS = {"distance_km": (0.5, 15), "prep_time_min": (5, 35), "traffic": (0, 2),
          "weather": (0, 2), "peak_hour": (0, 1), "courier_exp_years": (0, 8)}

def db():
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,
        created TEXT, distance_km REAL, prep_time_min REAL, traffic INT, weather INT,
        peak_hour INT, courier_exp_years REAL, predicted_min REAL)""")
    return c

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp

@app.route("/")
def home():
    return render_template("dashboard.html", m=METRICS)

@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json(silent=True) or {}
    vals = []
    for c in COLS:
        try:
            v = float(d[c])
        except (KeyError, TypeError, ValueError):
            return jsonify(error="Missing or invalid value: " + c), 400
        lo, hi = LIMITS[c]
        if not lo <= v <= hi:
            return jsonify(error="%s must be between %s and %s" % (c, lo, hi)), 400
        vals.append(v)
    minutes = round(max(8.0, float(MODEL.predict(pd.DataFrame([vals], columns=COLS))[0])), 1)
    with db() as c:
        c.execute("INSERT INTO predictions(created,distance_km,prep_time_min,traffic,weather,peak_hour,courier_exp_years,predicted_min) VALUES(?,?,?,?,?,?,?,?)",
                  [datetime.now().strftime("%d %b %H:%M")] + vals + [minutes])
    return jsonify(minutes=minutes, low=round(minutes - ERR), high=round(minutes + ERR))

@app.route("/api/history")
def history():
    with db() as c:
        rows = c.execute("SELECT created,distance_km,prep_time_min,traffic,weather,peak_hour,predicted_min FROM predictions ORDER BY id DESC LIMIT 10").fetchall()
    return jsonify(rows)

@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
