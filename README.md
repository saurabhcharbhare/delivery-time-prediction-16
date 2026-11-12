# Delivery Time Prediction (Machine Learning Project)
By Saurabh Charbhare

Flask backend + scikit-learn models. Predicts food delivery time (minutes) and saves every prediction in a SQLite database.

## Run on your computer
    pip install -r requirements.txt
    python generate_data.py     # creates delivery_data.csv (simulated data)
    python train.py             # trains 3 models, saves model.joblib, metrics.json and charts
    python app.py               # open http://127.0.0.1:5000

## Use a real dataset (recommended)
Download a food delivery dataset from Kaggle, make a CSV with exactly these columns and run `python train.py your_file.csv`:
distance_km, prep_time_min, traffic (0 low, 1 medium, 2 high), weather (0 clear, 1 rain, 2 fog), peak_hour (0/1), courier_exp_years, delivery_time_min

## API
- POST /api/predict   JSON body with the 6 inputs -> minutes and range
- GET  /api/history   last 10 predictions
- GET  /api/metrics   model comparison

## Deploy free on Render
1. Create a GitHub repository (name: delivery-time-prediction) and upload all files of this folder.
2. On render.com choose New > Blueprint (it reads render.yaml) or New > Web Service.
3. Build command: pip install -r requirements.txt && python generate_data.py && python train.py
4. Start command: gunicorn app:app
Note: the free plan sleeps when idle (first open takes about 30 seconds) and the SQLite history resets when the service restarts.
