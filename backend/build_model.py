import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json
import os
import re

# Predefined list of popular Bengaluru neighborhoods for cleaning location texts
NEIGHBORHOODS = [
    'Koramangala', 'Bellandur', 'Sarjapura', 'Frazer Town', 'Indiranagar', 
    'Nagarbhavi', 'HSR Layout', 'Basaveshwara Nagar', 'Shivaji Nagar', 
    'Upparpet', 'Vijayanagara', 'Whitefield', 'KR Puram', 'Cubbon Park',
    'Madiwala', 'Doddakannelli', 'Byatarayanapura', 'Pulikeshinagar',
    'Gandhinagar', 'Hebbal', 'Hennur', 'Jayanagar', 'Banashankari'
]

def extract_clean_region_name(location_series):
    """
    Analyzes address strings in a cluster to extract a clean road and area name.
    Example output: '18th Main Rd, Koramangala' or 'Sarjapura Rd, Doddakannelli'
    """
    # Clean nulls
    locations = location_series.dropna().tolist()
    if not locations:
        return "Unknown Hotspot"
        
    street_candidates = []
    detected_neighborhoods = []
    
    for loc in locations:
        # Split by comma
        parts = [p.strip() for p in loc.split(',')]
        if len(parts) > 0:
            # First part is usually the street/building name (e.g. 18th Main Road)
            street = parts[0]
            # Standardize abbreviations
            street = re.sub(r'(?i)road', 'Rd', street)
            street = re.sub(r'(?i)cross', 'Cross', street)
            street_candidates.append(street)
            
        # Scan for predefined neighborhood keywords in the whole string
        for nh in NEIGHBORHOODS:
            if re.search(r'(?i)\b' + re.escape(nh) + r'\b', loc):
                detected_neighborhoods.append(nh)
                break
                
    # Find most common street name
    most_common_street = "Main Road"
    if street_candidates:
        most_common_street = pd.Series(street_candidates).value_counts().index[0]
        # Clean up zip codes or long trailing numbers in street name
        most_common_street = re.sub(r'\b\d{5,6}\b', '', most_common_street).strip()
        
    # Find most common neighborhood
    if detected_neighborhoods:
        most_common_nh = pd.Series(detected_neighborhoods).value_counts().index[0]
        return f"{most_common_street}, {most_common_nh}"
        
    # Fallback: return the first two parts of the most common address
    if locations:
        parts = [p.strip() for p in pd.Series(locations).value_counts().index[0].split(',')]
        if len(parts) >= 2:
            return f"{parts[0]}, {parts[1]}"
        return parts[0]
        
    return "Unknown Area"


def run_training_pipeline():
    print("1. Loading raw dataset...")
    raw_csv_path = "../data/parking_violations_india.csv"
    if not os.path.exists(raw_csv_path):
        # Fallback if run from workspace directory instead of backend/
        raw_csv_path = "data/parking_violations_india.csv"
    if not os.path.exists(raw_csv_path):
        raw_csv_path = "parking_violations_india.csv"
        
    df = pd.read_csv(raw_csv_path)
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], format='mixed').dt.tz_convert('Asia/Kolkata')

    # Calculate target TOS weights
    print("2. Calculating Traffic Obstruction Score (TOS)...")
    def get_vehicle_weight(vt):
        vt_upper = str(vt).upper()
        if any(w in vt_upper for w in ['TANKER', 'BUS', 'TRUCK', 'LIVESTOCK', 'HEAVY']):
            return 5.0
        elif any(w in vt_upper for w in ['CAR', 'SUV', 'MAXI-CAB', 'TEMPO', 'JEEP']):
            return 3.0
        elif any(w in vt_upper for w in ['AUTO', 'THREE WHEELER', 'PASSENGER AUTO']):
            return 2.0
        else:
            return 1.0

    def get_violation_weight(violations):
        if not violations:
            return 2.0
        import ast
        try:
            parsed = ast.literal_eval(violations)
        except Exception:
            parsed = [x.strip('[]"\' ') for x in violations.split(',')]
        high_impact_keywords = ['CROSSING', 'MAIN ROAD', 'DOUBLE PARKING', 'FOOTPATH', 'CORNER']
        weight = 2.0
        for v in parsed:
            v_upper = str(v).upper()
            if any(kw in v_upper for kw in high_impact_keywords):
                weight = 4.0
                break
        return weight

    df['vehicle_weight'] = df['vehicle_type'].apply(get_vehicle_weight)
    df['violation_weight'] = df['violation_type'].apply(get_violation_weight)
    df['junction_multiplier'] = df['junction_name'].apply(lambda x: 1.5 if pd.notna(x) and x != 'No Junction' else 1.0)
    df['TOS'] = (df['vehicle_weight'] + df['violation_weight']) * df['junction_multiplier']

    # Spatial Clustering
    print("3. Running DBSCAN spatial clustering...")
    spatial_df = df.dropna(subset=['latitude', 'longitude']).copy()
    coords = spatial_df[['latitude', 'longitude']].values
    kms_per_radian = 6371.0088
    epsilon_meters = 150  # 150m radius threshold
    epsilon_rad = (epsilon_meters / 1000.0) / kms_per_radian

    db = DBSCAN(eps=epsilon_rad, min_samples=30, metric='haversine', algorithm='ball_tree')
    coords_rad = np.radians(coords)
    spatial_df['spot_id'] = db.fit_predict(coords_rad)

    # Filter noise
    hotspots_df = spatial_df[spatial_df['spot_id'] != -1].copy()
    
    # Calculate Centroids and Clean Region Names
    print("4. Calculating centroids and region names...")
    spots_meta = {}
    for spot_id in hotspots_df['spot_id'].unique():
        spot_rows = hotspots_df[hotspots_df['spot_id'] == spot_id]
        lat_mean = float(spot_rows['latitude'].mean())
        lon_mean = float(spot_rows['longitude'].mean())
        
        # Extract clean region name
        region_name = extract_clean_region_name(spot_rows['location'])
        
        # Most common police station in the cluster
        station = str(spot_rows['police_station'].mode().iloc[0] if not spot_rows['police_station'].mode().empty else 'Unknown')
        
        spots_meta[int(spot_id)] = {
            "spot_id": int(spot_id),
            "region_name": region_name,
            "latitude": lat_mean,
            "longitude": lon_mean,
            "police_station": station
        }

    # Save spots metadata
    with open("spots_meta.json", "w") as f:
        json.dump(spots_meta, f)
    print("Saved spots_meta.json successfully.")

    # Building Continuous Hourly Timeline
    print("5. Constructing hourly timeline for time-series...")
    hotspots_df['hour_bin'] = hotspots_df['created_datetime'].dt.floor('h')
    min_time = hotspots_df['hour_bin'].min()
    max_time = hotspots_df['hour_bin'].max()
    all_hours = pd.date_range(start=min_time, end=max_time, freq='h', tz='Asia/Kolkata')

    idx = pd.MultiIndex.from_product([hotspots_df['spot_id'].unique(), all_hours], names=['spot_id', 'hour_bin'])
    timeline_df = pd.DataFrame(index=idx).reset_index()

    hourly_agg = hotspots_df.groupby(['spot_id', 'hour_bin']).agg(
        violations_count=('id', 'count'),
        raw_tos=('TOS', 'sum')
    ).reset_index()

    timeline_df = pd.merge(timeline_df, hourly_agg, on=['spot_id', 'hour_bin'], how='left')
    timeline_df['violations_count'] = timeline_df['violations_count'].fillna(0)
    timeline_df['raw_tos'] = timeline_df['raw_tos'].fillna(0.0)

    # 6. Chronological splits
    print("6. Performing splits and feature engineering...")
    timeline_df = timeline_df.sort_values(by=['spot_id', 'hour_bin']).reset_index(drop=True)
    
    # Split threshold at 70% of total hours to calculate scaling index on train set only
    total_hours = len(all_hours)
    train_end_idx = int(total_hours * 0.70)
    train_cutoff = all_hours[train_end_idx]
    
    train_mask = timeline_df['hour_bin'] < train_cutoff
    tos_99 = timeline_df[train_mask]['raw_tos'].quantile(0.99)
    if tos_99 == 0:
        tos_99 = 1.0
        
    print(f"99th percentile raw TOS in Train set: {tos_99:.2f}")
    timeline_df['congestion_index'] = np.minimum(100.0, (timeline_df['raw_tos'] / tos_99) * 100.0)

    # Add temporal features
    timeline_df['hour_of_day'] = timeline_df['hour_bin'].dt.hour
    timeline_df['day_of_week'] = timeline_df['hour_bin'].dt.dayofweek
    timeline_df['is_weekend'] = (timeline_df['day_of_week'] >= 5).astype(int)

    # Shifting (prev_day_TOS, prev_week_TOS, rolling_TOS_24h)
    timeline_df['prev_day_TOS'] = timeline_df.groupby('spot_id')['congestion_index'].shift(24)
    timeline_df['prev_week_TOS'] = timeline_df.groupby('spot_id')['congestion_index'].shift(168)
    timeline_df['rolling_TOS_24h'] = timeline_df.groupby('spot_id')['congestion_index'].shift(1).rolling(24).mean()

    # Drop NaNs from timeline
    timeline_df = timeline_df.dropna(subset=['prev_week_TOS', 'rolling_TOS_24h']).copy()

    # Train-val-test cutoff points
    val_end_idx = int(total_hours * 0.85)
    val_cutoff = all_hours[val_end_idx]

    train_data = timeline_df[timeline_df['hour_bin'] < train_cutoff]
    
    # Calculate static target encodings based on Train Set only
    spot_train_avg = train_data.groupby('spot_id')['congestion_index'].mean().to_dict()
    timeline_df['spot_avg_TOS'] = timeline_df['spot_id'].map(spot_train_avg).fillna(0.0)

    spot_hw_train_avg = train_data.groupby(['spot_id', 'hour_of_day', 'day_of_week'])['congestion_index'].mean().reset_index()
    spot_hw_train_avg.rename(columns={'congestion_index': 'spot_hour_weekday_avg_TOS'}, inplace=True)

    timeline_df = pd.merge(timeline_df, spot_hw_train_avg, on=['spot_id', 'hour_of_day', 'day_of_week'], how='left')
    timeline_df['spot_hour_weekday_avg_TOS'] = timeline_df['spot_hour_weekday_avg_TOS'].fillna(timeline_df['spot_avg_TOS'])

    # Save timeline_data.csv for Flask usage
    timeline_df.to_csv("timeline_data.csv", index=False)
    print("Saved timeline_data.csv successfully.")

    # 7. Model Training
    print("7. Training Random Forest Model...")
    features = [
        'hour_of_day', 'day_of_week', 'is_weekend', 
        'spot_avg_TOS',
        'prev_day_TOS', 'prev_week_TOS', 'rolling_TOS_24h'
    ]
    target = 'congestion_index'

    train_split = timeline_df[timeline_df['hour_bin'] < train_cutoff]
    val_split = timeline_df[(timeline_df['hour_bin'] >= train_cutoff) & (timeline_df['hour_bin'] < val_cutoff)]
    test_split = timeline_df[timeline_df['hour_bin'] >= val_cutoff]

    X_train, y_train = train_split[features], train_split[target]
    X_val, y_val = val_split[features], val_split[target]
    X_test, y_test = test_split[features], test_split[target]

    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Print Robust Validation Comparative Metrics
    print("\n--- ML Pipeline Evaluation Summary ---")
    for name, X, y in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        preds = model.predict(X)
        preds_clipped = np.clip(preds, 0.0, 100.0)
        preds_thresh = np.where(preds_clipped < 5.0, 0.0, preds_clipped)
        
        # Zero baseline
        mae_zero = mean_absolute_error(y, np.zeros_like(y))
        
        # Raw model
        mae_raw = mean_absolute_error(y, preds_clipped)
        acc_2_raw = np.mean(np.abs(y - preds_clipped) < 2.0) * 100
        
        # Thresholded model
        mae_thresh = mean_absolute_error(y, preds_thresh)
        acc_2_thresh = np.mean(np.abs(y - preds_thresh) < 2.0) * 100
        
        print(f"  {name:10} Set:")
        print(f"    Zero-Baseline MAE: {mae_zero:.2f}%")
        print(f"    Raw Model MAE:     {mae_raw:.2f}% (Acc@2%: {acc_2_raw:.2f}%)")
        print(f"    Thresholded MAE:   {mae_thresh:.2f}% (Acc@2%: {acc_2_thresh:.2f}%)")


    # Serialize Model & parameters
    model_meta = {
        "features": features,
        "target": target,
        "tos_99": float(tos_99),
        "spot_train_avg": spot_train_avg
    }
    
    joblib.dump(model, "model.joblib")
    joblib.dump(model_meta, "model_meta.joblib")
    print("Model and metadata serialized successfully.")

if __name__ == "__main__":
    run_training_pipeline()
