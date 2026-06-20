# Traffic Hotspots Predictive Command Center 🚨

An AI-driven traffic obstruction forecasting dashboard and command center built for police departments. The application leverages unsupervised spatial clustering (DBSCAN) and a supervised Random Forest Regressor to forecast hourly parking-induced Congestion Risk Index scores (0-100%) for critical hotspots in Bengaluru, India. It routes police patrol resources dynamically using real-time predictive insights.

---

## 🖥️ Command Center Interface Preview

* **Interactive Map**: Renders Leaflet-based hotspot markers colored dynamically by risk level (Green: Low, Yellow: Moderate, Orange: High, Red/Pulse: Critical).
* **Priority Hotspots Queue**: Ranked queue of the top 20 critical hotspots in the selected jurisdiction.
* **Live KPI Metrics**: Displays dynamic statistics including Average Risk Index, Critical Choke Zones, and Total Active Violations.
* **Congestion Analytics**: Real-time projected hourly risk profiles and vehicle breakdown charts.
* **Patrol Routing Overlay**: Provides precise action recommendations (e.g., dispatching towing trucks or issuing patrol warnings) when hotspots are clicked.

---

## 🏗️ Technical Architecture

The project is split into three main components:

1. **Jupyter Notebook (`Poor_Visibility_and_Parking_induced_congestion.ipynb`)**: 
   Contains the complete research pipeline including exploratory data analysis, spatial clustering, time-series timeline reconstruction, feature engineering, RF training, and metric evaluations.
2. **Flask Backend API (`backend/`)**:
   Provides REST endpoints for serving live predictions, retrieving historical timelines, triggering retraining, and ingesting daily logs.
3. **Vite React Frontend (`frontend/`)**:
   A dark-themed, glassmorphic single-page dashboard displaying the interactive map, controls, queues, and Chart.js analytics.

---

## 🧠 Machine Learning Pipeline

### 1. Spatial Hotspot Clustering (DBSCAN)
* **Objective**: Cluster raw coordinates of historical violations into dense parking hotspots.
* **Parameters**: 150-meter radius (`epsilon`), minimum 30 violations (`min_samples`), utilizing the Haversine metric.
* **Output**: Identifies **267 distinct hotspots** across Bengaluru, mapping every raw violation to a canonical `spot_id`.

### 2. Hourly Timeline Reconstruction & Normalization
* **Objective**: Aggregate raw violations chronologically to forecast hourly risk.
* **Target variable**: Sum of raw Traffic Obstruction Scores (TOS) per spot per hour. Raw TOS weights vehicle types (commercial/heavy = 5, cars = 3, autos = 2) and violation impacts (footpath/corner blockages = 4, others = 2) multiplied by a junction factor (1.5).
* **Congestion Risk Index (0-100%)**: Normalized by dividing the raw hourly TOS by the 99th percentile of raw training set TOS.

### 3. Feature Engineering & Chronological Splits
* **Temporal features**: `hour_of_day`, `day_of_week`, `is_weekend`.
* **Dynamic Lags**: Lags are calculated using timezone-aware (IST) shifts to prevent lookahead bias:
  * `prev_day_TOS` (risk 24 hours ago)
  * `prev_week_TOS` (risk 168 hours ago)
  * `rolling_TOS_24h` (rolling average of the previous 24 hours)
* **Train/Val/Test Split**: Chronological partition (70% Train, 15% Val, 15% Test) to ensure evaluation is strictly out-of-sample in time.

### 4. Zero-Inflation & Hurdle Thresholding
Because parking violations are discrete events, **96.32% of the spot-hour timeline has exactly 0 violations**. A raw regressor predicting small values (1-3% risk) on quiet hours results in high baseline errors.
* **5.0% Hurdle Threshold**: Applied at inference time—any prediction below 5.0% risk is automatically rounded to 0.0%.
* **Test Set Performance Results**:
  * **Zero-Prediction Baseline MAE**: `1.78%`
  * **Raw Regressor MAE**: `2.09%` (Accuracy within 2% tolerance: `85.39%`)
  * **Hurdle-Thresholded Model MAE**: **`1.67%`** (Outperforms the zero-baseline! Accuracy within 2% tolerance: **`91.91%`**)
  * **Active Hours MAE**: **`22.8%`** (Reduced from the zero-baseline error of `49.9%` when congestion is actually active).

---

## 📂 Project Directory Structure

```text
├── Poor_Visibility_and_Parking_induced_congestion.ipynb  # Main ML Research Notebook
├── README.md                                             # Project Overview & Setup
├── data/
│   └── parking_violations_india.csv                      # Raw violations dataset (Git LFS)
├── backend/
│   ├── app.py                                            # Flask Server Web API
│   ├── build_model.py                                    # Model training and generation script
│   ├── requirements.txt                                  # Backend Python dependencies
│   ├── spots_meta.json                                   # Spot centroids & names metadata
│   ├── model.joblib                                      # Trained Random Forest Regressor
│   ├── model_meta.joblib                                 # Model feature list and train meta
│   └── timeline_data.csv                                 # Generated hourly forecasting timeline
└── frontend/
    ├── package.json                                      # Node dependencies
    ├── vite.config.js                                    # Vite configuration
    ├── index.html                                        # Application root page
    ├── src/
    │   ├── main.jsx                                      # React App entry point
    │   ├── App.jsx                                       # Primary Command Center Dashboard UI
    │   └── index.css                                     # Core layout styles & scrollbar behaviors
    └── public/                                           # Static assets
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
Ensure you have the following installed on your machine:
* Python 3.9+
* Node.js v18+ (npm)

---

### 2. Backend Setup
Navigate to the `backend/` directory, create a virtual environment, install dependencies, and build the model:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the training script to cluster data, compile timeline, and serialize the model
python build_model.py

# Start the Flask backend server
python app.py
```
*The Flask server will start listening on `http://localhost:5000`.*

---

### 3. Frontend Setup
Open a new terminal session, navigate to the `frontend/` directory, install Node packages, and start Vite:

```bash
cd frontend
npm install

# Run the dev server
npm run dev
```
*The dev server will launch. Open **`http://localhost:5173/`** (or the port specified in terminal) in your browser.*

---

## 🔍 Validation Checking & Boundary Constraints
* **Active Database Window**: The operational data bounds span from **`2024-03-17`** to **`2024-04-08`**.
* **Insufficient Data Validation**: If a date picker date is selected outside this range, the dashboard remains active but shows:
  * In the **Hotspots Queue**: A clean inline indicator explaining why predictions are unavailable.
  * In the **Map Center**: Clears all markers and displays a non-blocking floating header card warning `⚠️ MAP CLEAR: INSUFFICIENT DATA`.
  * Changing the date back to a valid range instantly restores predictions and rendering.
