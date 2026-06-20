from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
import os
import threading
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# Global variables to hold data in memory
model = None
model_meta = None
spots_meta = None
timeline_df = None
is_training = False

# Load models and data at startup
def load_resources():
    global model, model_meta, spots_meta, timeline_df
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.joblib")
        meta_path = os.path.join(base_dir, "model_meta.joblib")
        spots_path = os.path.join(base_dir, "spots_meta.json")
        timeline_path = os.path.join(base_dir, "timeline_data.csv")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_meta = joblib.load(meta_path)
            print("Model and metadata loaded.")
        else:
            print(f"WARNING: {model_path} not found. Run build_model.py first!")

        if os.path.exists(spots_path):
            with open(spots_path, "r") as f:
                spots_meta = json.load(f)
            print("Spots metadata loaded.")
        else:
            print(f"WARNING: {spots_path} not found.")

        if os.path.exists(timeline_path):
            dtypes = {
                'spot_id': 'int16',
                'violations_count': 'int16',
                'raw_tos': 'float32',
                'congestion_index': 'float32',
                'hour_of_day': 'int8',
                'day_of_week': 'int8',
                'is_weekend': 'int8',
                'prev_day_TOS': 'float32',
                'prev_week_TOS': 'float32',
                'rolling_TOS_24h': 'float32',
                'spot_avg_TOS': 'float32',
                'spot_hour_weekday_avg_TOS': 'float32'
            }
            timeline_df = pd.read_csv(timeline_path, dtype=dtypes)
            timeline_df['hour_bin'] = pd.to_datetime(timeline_df['hour_bin'])
            print("Timeline data loaded with optimized memory footprint.")
        else:
            print(f"WARNING: {timeline_path} not found.")
    except Exception as e:
        print(f"Error loading resources: {e}")

# Load resources at startup so they are available when imported by WSGI servers like Gunicorn
load_resources()

# Helper: Find closest hotspot centroid within 150m (0.00135 degrees approx)
def allocate_to_nearest_spot(lat, lon):
    if not spots_meta:
        return -1
    min_dist = float('inf')
    closest_spot_id = -1
    # Very basic Euclidean degree approximation (fine for small local distance comparison)
    for spot_id, spot in spots_meta.items():
        dist = np.sqrt((lat - spot['latitude'])**2 + (lon - spot['longitude'])**2)
        if dist < min_dist:
            min_dist = dist
            closest_spot_id = int(spot_id)
            
    # Proximity threshold: 150 meters is ~0.00135 degrees
    if min_dist <= 0.00135:
        return closest_spot_id
    return -1

# API: Get Hotspots list
@app.route("/api/spots", methods=["GET"])
def get_spots():
    if not spots_meta:
        return jsonify({"error": "Metadata not loaded"}), 500
    return jsonify(spots_meta)

# Helper: compute batch predictions for a localized query datetime
def compute_predictions_raw(query_dt):
    global model, model_meta, timeline_df, spots_meta
    if model is None or timeline_df is None or spots_meta is None:
        return None

    hour_of_day = query_dt.hour
    day_of_week = query_dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0

    # Pre-filter timeline_df for fast vectorized lookup
    prev_day_dt = query_dt - pd.Timedelta(days=1)
    prev_week_dt = query_dt - pd.Timedelta(days=7)
    rolling_start = query_dt - pd.Timedelta(hours=24)
    rolling_end = query_dt - pd.Timedelta(hours=1)

    # Filter by timestamps once
    query_rows = timeline_df[timeline_df['hour_bin'] == query_dt]
    prev_day_rows = timeline_df[timeline_df['hour_bin'] == prev_day_dt]
    prev_week_rows = timeline_df[timeline_df['hour_bin'] == prev_week_dt]
    
    # Pre-filter rolling window rows
    rolling_rows_all = timeline_df[
        (timeline_df['hour_bin'] >= rolling_start) & 
        (timeline_df['hour_bin'] <= rolling_end)
    ]
    
    # Filter for spot_hour_weekday_avg
    hw_rows_all = timeline_df[
        (timeline_df['hour_of_day'] == hour_of_day) & 
        (timeline_df['day_of_week'] == day_of_week)
    ]

    # Convert to fast dictionaries for O(1) lookups
    query_dict = query_rows.set_index('spot_id')[['congestion_index', 'violations_count']].to_dict('index')
    prev_day_dict = prev_day_rows.set_index('spot_id')['congestion_index'].to_dict()
    prev_week_dict = prev_week_rows.set_index('spot_id')['congestion_index'].to_dict()
    rolling_dict = rolling_rows_all.groupby('spot_id')['congestion_index'].mean().to_dict()
    hw_dict = hw_rows_all.drop_duplicates(subset=['spot_id']).set_index('spot_id')['spot_hour_weekday_avg_TOS'].to_dict()

    features_data = []
    spots_list = list(spots_meta.items())

    for spot_id_str, spot in spots_list:
        spot_id = int(spot_id_str)
        spot_avg = model_meta['spot_train_avg'].get(spot_id, 0.0)
        
        spot_hw_avg = hw_dict.get(spot_id, spot_avg)
        prev_day_val = prev_day_dict.get(spot_id, spot_hw_avg)
        prev_week_val = prev_week_dict.get(spot_id, spot_hw_avg)
        rolling_val = rolling_dict.get(spot_id, spot_hw_avg)
        
        features_data.append([
            hour_of_day, day_of_week, is_weekend,
            spot_avg,
            prev_day_val, prev_week_val, rolling_val
        ])

    # Batch Predict in one call (vectorized)
    feature_df = pd.DataFrame(features_data, columns=model_meta['features'])
    predicted_scores = model.predict(feature_df)

    results = []
    for idx, (spot_id_str, spot) in enumerate(spots_list):
        spot_id = int(spot_id_str)
        pred_score = max(0.0, min(100.0, float(predicted_scores[idx])))
        if pred_score < 5.0:
            pred_score = 0.0
        
        # Check actuals
        q_data = query_dict.get(spot_id, {})
        actual_score = q_data.get('congestion_index', None)
        violations_count = q_data.get('violations_count', 0)

        results.append({
            "spot_id": spot_id,
            "region_name": spot['region_name'],
            "latitude": spot['latitude'],
            "longitude": spot['longitude'],
            "police_station": spot['police_station'],
            "predicted_congestion": pred_score,
            "actual_congestion": actual_score,
            "violations_count": int(violations_count)
        })
    return results

# API: Dynamic hourly predictions for a given date/time
@app.route("/api/predict", methods=["GET"])
def predict_hour():
    dt_str = request.args.get("datetime") # Expects format "YYYY-MM-DDTHH:00:00"
    if not dt_str:
        return jsonify({"error": "Missing 'datetime' parameter"}), 400
        
    try:
        # Parse query datetime (ensure localized to IST)
        query_dt = pd.to_datetime(dt_str).tz_localize(None).tz_localize('Asia/Kolkata')
    except Exception as e:
        return jsonify({"error": f"Invalid datetime format: {e}"}), 400

    # Boundary check
    min_date = pd.Timestamp("2024-03-17 00:00:00").tz_localize('Asia/Kolkata')
    max_date = pd.Timestamp("2024-04-08 23:00:00").tz_localize('Asia/Kolkata')
    
    if query_dt < min_date or query_dt > max_date:
        return jsonify({
            "status": "error",
            "error_type": "INSUFFICIENT_DATA",
            "message": "Reliable forecasts require preceding 24h and 7d violation logs. Active database records only span from 17-03-2024 to 08-04-2024."
        }), 200

    results = compute_predictions_raw(query_dt)
    if results is None:
        return jsonify({"error": "Model or timeline data not loaded"}), 500

    return jsonify({
        "timestamp": query_dt.isoformat(),
        "predictions": results
    })

# API: Get historical details for a single hotspot (last 7 days of predictions vs actuals)
@app.route("/api/history", methods=["GET"])
def get_history():
    global timeline_df, model, model_meta
    if timeline_df is None or model is None:
        return jsonify({"error": "Timeline data or model not loaded"}), 500
        
    spot_id_str = request.args.get("spot_id")
    if not spot_id_str:
        return jsonify({"error": "Missing 'spot_id' parameter"}), 400
        
    spot_id = int(spot_id_str)
    
    # Filter timeline for last 7 days of dataset
    max_time = timeline_df['hour_bin'].max()
    min_time = max_time - pd.Timedelta(days=7)
    
    spot_timeline = timeline_df[
        (timeline_df['spot_id'] == spot_id) & 
        (timeline_df['hour_bin'] >= min_time)
    ].sort_values(by='hour_bin').copy()
    
    if spot_timeline.empty:
        return jsonify({"spot_id": spot_id, "history": []})
        
    # Prepare batch features (vectorized)
    features_data = []
    for idx, row in spot_timeline.iterrows():
        features_data.append([
            int(row['hour_of_day']), int(row['day_of_week']), int(row['is_weekend']),
            float(row['spot_avg_TOS']),
            float(row['prev_day_TOS']), float(row['prev_week_TOS']), float(row['rolling_TOS_24h'])
        ])
        
    feature_df = pd.DataFrame(features_data, columns=model_meta['features'])
    predicted_scores = model.predict(feature_df)
    
    history_records = []
    for idx, (_, row) in enumerate(spot_timeline.iterrows()):
        pred_score = max(0.0, min(100.0, float(predicted_scores[idx])))
        if pred_score < 5.0:
            pred_score = 0.0
        history_records.append({
            "timestamp": row['hour_bin'].isoformat(),
            "actual_congestion": float(row['congestion_index']),
            "predicted_congestion": pred_score,
            "violations_count": int(row['violations_count'])
        })
        
    return jsonify({
        "spot_id": spot_id,
        "history": history_records
    })

# API: Ingest daily CSV data
@app.route("/api/ingest", methods=["POST"])
def ingest_data():
    global timeline_df, model_meta
    if timeline_df is None:
        return jsonify({"error": "Timeline data not loaded"}), 500
        
    payload = request.json # Expects a list of violation JSON objects
    if not payload or not isinstance(payload, list):
        return jsonify({"error": "Payload must be a list of violation objects"}), 400
        
    # Process new violations
    print(f"Ingesting {len(payload)} new violations...")
    
    # Define weight calculator weights
    tos_99 = model_meta.get("tos_99", 33.0)
    
    # We aggregate violations into hourly bins per spot
    new_violations_grouped = {}
    
    for v in payload:
        lat = float(v.get('latitude', 0))
        lon = float(v.get('longitude', 0))
        created_str = v.get('created_datetime')
        vehicle_type = v.get('vehicle_type', 'SCOOTER')
        violation_type = v.get('violation_type', '["NO PARKING"]')
        junction_name = v.get('junction_name', 'No Junction')
        police_station = v.get('police_station', 'Unknown')
        
        # 1. Allocate to nearest cluster spot
        spot_id = allocate_to_nearest_spot(lat, lon)
        if spot_id == -1:
            continue # Skip noise points
            
        # 2. Calculate individual TOS
        # Vehicle Weight
        vt_upper = str(vehicle_type).upper()
        if any(w in vt_upper for w in ['TANKER', 'BUS', 'TRUCK', 'HEAVY']):
            vf = 5.0
        elif any(w in vt_upper for w in ['CAR', 'SUV', 'MAXI-CAB', 'TEMPO', 'JEEP']):
            vf = 3.0
        elif any(w in vt_upper for w in ['AUTO', 'THREE WHEELER', 'PASSENGER AUTO']):
            vf = 2.0
        else:
            vf = 1.0
            
        # Violation Weight
        high_impact_keywords = ['CROSSING', 'MAIN ROAD', 'DOUBLE PARKING', 'FOOTPATH', 'CORNER']
        sv = 2.0
        if isinstance(violation_type, list):
            violations_list = violation_type
        else:
            import ast
            try:
                violations_list = ast.literal_eval(violation_type)
            except Exception:
                violations_list = [x.strip('[]"\' ') for x in violation_type.split(',')]
        for vt_item in violations_list:
            if any(kw in str(vt_item).upper() for kw in high_impact_keywords):
                sv = 4.0
                break
                
        # Junction multiplier
        jm = 1.5 if pd.notna(junction_name) and junction_name != 'No Junction' else 1.0
        tos = (vf + sv) * jm
        
        # 3. Parse timestamp and round to hour
        dt = pd.to_datetime(created_str).tz_localize(None).tz_localize('Asia/Kolkata')
        hour_bin = dt.floor('h')
        
        key = (spot_id, hour_bin)
        if key not in new_violations_grouped:
            new_violations_grouped[key] = {"count": 0, "tos_sum": 0.0}
            
        new_violations_grouped[key]["count"] += 1
        new_violations_grouped[key]["tos_sum"] += tos

    # Update the core timeline data
    records_added = 0
    for (spot_id, hour_bin), data in new_violations_grouped.items():
        # Check if record already exists
        existing_idx = timeline_df[
            (timeline_df['spot_id'] == spot_id) & 
            (timeline_df['hour_bin'] == hour_bin)
        ].index
        
        raw_tos = data["tos_sum"]
        violations_count = data["count"]
        congestion_index = min(100.0, (raw_tos / tos_99) * 100.0)
        
        if not existing_idx.empty:
            # Overwrite/Add to existing
            idx_val = existing_idx[0]
            timeline_df.at[idx_val, 'violations_count'] += violations_count
            new_raw_tos = timeline_df.at[idx_val, 'raw_tos'] + raw_tos
            timeline_df.at[idx_val, 'raw_tos'] = new_raw_tos
            timeline_df.at[idx_val, 'congestion_index'] = min(100.0, (new_raw_tos / tos_99) * 100.0)
        else:
            # Create a new row
            new_row = {
                "spot_id": spot_id,
                "hour_bin": hour_bin,
                "violations_count": violations_count,
                "raw_tos": raw_tos,
                "congestion_index": congestion_index,
                "hour_of_day": hour_bin.hour,
                "day_of_week": hour_bin.dayofweek,
                "is_weekend": 1 if hour_bin.dayofweek >= 5 else 0
            }
            # Temporary dataframe and concat
            new_row_df = pd.DataFrame([new_row])
            timeline_df = pd.concat([timeline_df, new_row_df], ignore_index=True)
            records_added += 1

    # Recalculate lags for the updated spots
    print("Recalculating lags for timeline...")
    timeline_df = timeline_df.sort_values(by=['spot_id', 'hour_bin']).reset_index(drop=True)
    timeline_df['prev_day_TOS'] = timeline_df.groupby('spot_id')['congestion_index'].shift(24)
    timeline_df['prev_week_TOS'] = timeline_df.groupby('spot_id')['congestion_index'].shift(168)
    timeline_df['rolling_TOS_24h'] = timeline_df.groupby('spot_id')['congestion_index'].shift(1).rolling(24).mean()

    # Save to disk
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timeline_path = os.path.join(base_dir, "timeline_data.csv")
    timeline_df.to_csv(timeline_path, index=False)
    print("timeline_data.csv updated on disk.")
    
    return jsonify({
        "status": "success",
        "message": f"Successfully ingested daily logs. {records_added} new timeline hours generated."
    })

# Retraining Thread function
def retrain_model_thread():
    global is_training
    try:
        from build_model import run_training_pipeline
        run_training_pipeline()
        load_resources() # Reload weights and data
        print("Retraining completed successfully and resources reloaded.")
    except Exception as e:
        print(f"Error during background retraining: {e}")
    finally:
        is_training = False

# API: Trigger Model Retraining
@app.route("/api/retrain", methods=["POST"])
def trigger_retrain():
    global is_training
    if is_training:
        return jsonify({"status": "running", "message": "Model is already retraining."}), 400
        
    is_training = True
    thread = threading.Thread(target=retrain_model_thread)
    thread.start()
    return jsonify({"status": "training", "message": "Model retraining started in the background."})

# API: Check training status
@app.route("/api/retrain/status", methods=["GET"])
def check_retrain_status():
    global is_training
    return jsonify({"is_training": is_training})

# API: General dashboard statistics (dynamic for datetime and jurisdiction)
@app.route("/api/stats", methods=["GET"])
def get_stats():
    global timeline_df, spots_meta
    if timeline_df is None or spots_meta is None:
        return jsonify({"error": "Timeline data not loaded"}), 500
        
    dt_str = request.args.get("datetime")
    police_station = request.args.get("police_station", "ALL")

    if dt_str:
        try:
            query_dt = pd.to_datetime(dt_str).tz_localize(None).tz_localize('Asia/Kolkata')
        except Exception as e:
            return jsonify({"error": f"Invalid datetime format: {e}"}), 400
    else:
        max_time_val = timeline_df['hour_bin'].max()
        query_dt = pd.to_datetime(max_time_val).tz_localize(None).tz_localize('Asia/Kolkata')

    # Boundary check for timeline data range
    min_date = pd.Timestamp("2024-03-17 00:00:00").tz_localize('Asia/Kolkata')
    max_date = pd.Timestamp("2024-04-08 23:00:00").tz_localize('Asia/Kolkata')
    
    if query_dt < min_date or query_dt > max_date:
        return jsonify({
            "status": "error",
            "error_type": "INSUFFICIENT_DATA",
            "message": "Reliable forecasts require preceding 24h and 7d violation logs. Active database records only span from 17-03-2024 to 08-04-2024."
        }), 200

    predictions_list = compute_predictions_raw(query_dt)
    if predictions_list is None:
        return jsonify({"error": "Model or timeline data not loaded"}), 500

    # Filter predictions list by jurisdiction if specified
    if police_station and police_station != "ALL" and police_station != "null":
        predictions_list = [p for p in predictions_list if p['police_station'] == police_station]
        station_spots = [int(spot_id) for spot_id, spot in spots_meta.items() if spot['police_station'] == police_station]
        
        last_day_mask = (timeline_df['hour_bin'] <= query_dt) & (timeline_df['hour_bin'] > (query_dt - pd.Timedelta(days=1))) & (timeline_df['spot_id'].isin(station_spots))
        day_df = timeline_df[last_day_mask]
    else:
        last_day_mask = (timeline_df['hour_bin'] <= query_dt) & (timeline_df['hour_bin'] > (query_dt - pd.Timedelta(days=1)))
        day_df = timeline_df[last_day_mask]

    avg_congestion = float(np.mean([p['predicted_congestion'] for p in predictions_list])) if predictions_list else 0.0
    active_hotspots = int(sum(1 for p in predictions_list if p['predicted_congestion'] >= 75))
    total_violations = int(day_df['violations_count'].sum()) if not day_df.empty else 0
    
    return jsonify({
        "last_updated": query_dt.isoformat(),
        "average_congestion_risk": avg_congestion,
        "active_critical_hotspots": active_hotspots,
        "total_day_violations": total_violations
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
