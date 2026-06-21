import React, { useState, useEffect, useRef } from 'react';
import L from 'leaflet';
import { Chart, registerables } from 'chart.js';

// Register all Chart.js components
Chart.register(...registerables);

// Base API URL pointing to the Flask Backend (resolved dynamically in production)
const rawApiUrl = import.meta.env.VITE_API_URL || "http://localhost:5000";
const API_URL = rawApiUrl.endsWith('/api') ? rawApiUrl : `${rawApiUrl}/api`;

// MapmyIndia / Mappls API Key configuration (loads proprietary mapping infrastructure)
const MAPMYINDIA_KEY = import.meta.env.VITE_MAPMYINDIA_API_KEY || "";

export default function App() {
    // 1. Theme and Core states
    const [theme, setTheme] = useState("dark");
    const [activeDate, setActiveDate] = useState("2024-04-05"); // default target test date
    const [activeHour, setActiveHour] = useState(10); // default morning peak hour (10:00 AM)
    const [selectedJurisdiction, setSelectedJurisdiction] = useState("ALL");
    const [spots, setSpots] = useState({});
    const [predictions, setPredictions] = useState([]);
    const [stats, setStats] = useState({
        average_congestion_risk: 0,
        active_critical_hotspots: 0,
        total_day_violations: 0
    });
    
    // 2. Play Loop & Status
    const [isPlaying, setIsPlaying] = useState(false);
    const [isTraining, setIsTraining] = useState(false);
    const [selectedSpotId, setSelectedSpotId] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [insufficientDataError, setInsufficientDataError] = useState(null);

    // 3. Map & Chart Refs
    const mapContainerRef = useRef(null);
    const mapRef = useRef(null);
    const tileLayerRef = useRef(null);
    const markerGroupRef = useRef(null);
    
    const mainChartRef = useRef(null);
    const mainChartInstance = useRef(null);
    const hourlyChartRef = useRef(null);
    const hourlyChartInstance = useRef(null);
    const vehicleChartRef = useRef(null);
    const vehicleChartInstance = useRef(null);

    // Play loop timer ref
    const playTimerRef = useRef(null);

    // --- EFFECT A: LOAD HOTSPOTS CENTROIDS ---
    useEffect(() => {
        const fetchSpots = async () => {
            try {
                const res = await fetch(`${API_URL}/spots`);
                if (res.ok) {
                    const data = await res.json();
                    setSpots(data);
                }
            } catch (err) {
                console.error("Error fetching spots centroids:", err);
            }
        };
        fetchSpots();
    }, []);

    // --- EFFECT B: INITIALIZE LEAFLET MAP ---
    useEffect(() => {
        if (!mapRef.current && mapContainerRef.current) {
            // Initialize map centered on Bengaluru
            const mapInstance = L.map(mapContainerRef.current, {
                zoomControl: true,
                attributionControl: false
            }).setView([12.9716, 77.5946], 12);

            mapRef.current = mapInstance;

            // Create marker group layer
            markerGroupRef.current = L.layerGroup().addTo(mapInstance);
        }
    }, []);

    // --- EFFECT C: TOGGLE MAP TILE THEME DYNAMICALLY ---
    useEffect(() => {
        if (mapRef.current) {
            // Remove old tile layer if exists
            if (tileLayerRef.current) {
                mapRef.current.removeLayer(tileLayerRef.current);
            }

            // Define tile URLs and attributions
            let tileUrl = "";
            let attribution = "";

            if (MAPMYINDIA_KEY) {
                // MapmyIndia / Mappls Raster Tiles integration (resolves to standard PNG tiles)
                tileUrl = `https://apis.mappls.com/advancedmaps/v1/${MAPMYINDIA_KEY}/raster/map_style/{z}/{x}/{y}.png`;
                attribution = "© MapmyIndia";
            } else {
                // Standard CartoDB maps (fallback)
                tileUrl = theme === "dark" 
                    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    : "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png";
                attribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>';
            }

            const newTileLayer = L.tileLayer(tileUrl, { 
                maxZoom: 20,
                attribution: attribution
            });
            newTileLayer.addTo(mapRef.current);
            tileLayerRef.current = newTileLayer;
        }
        
        // Update document theme attribute
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    // --- EFFECT D: FETCH PREDICTIONS & STATS ON TIME/FILTER CHANGE ---
    useEffect(() => {
        const fetchData = async () => {
            setIsLoading(true);
            const dtParam = `${activeDate}T${String(activeHour).padStart(2, '0')}:00:00`;
            try {
                // Fetch predictions
                const predRes = await fetch(`${API_URL}/predict?datetime=${dtParam}`);
                if (predRes.ok) {
                    const data = await predRes.json();
                    if (data.status === "error" && data.error_type === "INSUFFICIENT_DATA") {
                        setInsufficientDataError(data.message);
                        setPredictions([]);
                        setStats({
                            average_congestion_risk: 0,
                            active_critical_hotspots: 0,
                            total_day_violations: 0
                        });
                        setIsLoading(false);
                        return;
                    } else {
                        setInsufficientDataError(null);
                        setPredictions(data.predictions);
                    }
                }

                // Fetch stats passing target datetime and current jurisdiction filter
                const statsRes = await fetch(`${API_URL}/stats?datetime=${dtParam}&police_station=${selectedJurisdiction}`);
                if (statsRes.ok) {
                    const statsData = await statsRes.json();
                    if (statsData.status === "error" && statsData.error_type === "INSUFFICIENT_DATA") {
                        // Handled by predictions check
                    } else {
                        setStats(statsData);
                    }
                }
            } catch (err) {
                console.error("Error fetching prediction payload:", err);
            } finally {
                setIsLoading(false);
            }
        };

        fetchData();
    }, [activeDate, activeHour, selectedJurisdiction]);

    // --- EFFECT E: RENDER MAP MARKERS ---
    useEffect(() => {
        if (mapRef.current && markerGroupRef.current && predictions.length > 0) {
            markerGroupRef.current.clearLayers();

            // Filter by police station
            const filteredPreds = predictions.filter(p => 
                selectedJurisdiction === "ALL" || p.police_station === selectedJurisdiction
            );

            filteredPreds.forEach(p => {
                const score = p.predicted_congestion;
                
                // Color scale
                let color = "#00ff87"; // Green
                if (score >= 75) color = "#ff3838"; // Red
                else if (score >= 50) color = "#e67e22"; // Orange
                else if (score >= 25) color = "#f1c40f"; // Yellow

                const radius = 6 + (score * 0.18);

                const markerOptions = {
                    radius: radius,
                    fillColor: color,
                    color: color,
                    weight: 1.5,
                    opacity: 0.8,
                    fillOpacity: 0.45,
                    className: score >= 75 ? 'map-pulse-marker' : ''
                };

                const marker = L.circleMarker([p.latitude, p.longitude], markerOptions);
                
                marker.bindTooltip(`
                    <div style="font-family:'Outfit',sans-serif; font-size:11px; padding:2px;">
                        <strong>${p.region_name}</strong><br/>
                        Station: <strong>${p.police_station.toUpperCase()}</strong><br/>
                        Risk: <strong style="color:${color}">${score.toFixed(1)}%</strong>
                    </div>
                `, { sticky: true });

                marker.on("click", () => {
                    setSelectedSpotId(p.spot_id);
                });

                marker.addTo(markerGroupRef.current);
            });
        }
    }, [predictions, selectedJurisdiction]);

    // --- EFFECT F: SIMULATION PLAY/PAUSE LOOP ---
    useEffect(() => {
        if (isPlaying) {
            playTimerRef.current = setInterval(() => {
                setActiveHour(prev => (prev + 1) % 24);
            }, 1000);
        } else {
            clearInterval(playTimerRef.current);
        }

        return () => clearInterval(playTimerRef.current);
    }, [isPlaying]);

    // --- EFFECT G: CHECK RETRAINING STATUS ON POLL ---
    useEffect(() => {
        let pollTimer = null;
        if (isTraining) {
            pollTimer = setInterval(async () => {
                try {
                    const res = await fetch(`${API_URL}/retrain/status`);
                    if (res.ok) {
                        const data = await res.json();
                        if (!data.is_training) {
                            setIsTraining(false);
                            alert("AI Model successfully retrained on newly ingested daily logs!");
                            // Trigger page data reload
                            setActiveHour(prev => prev);
                        }
                    }
                } catch (err) {
                    console.error("Error checking train status:", err);
                }
            }, 2000);
        }
        return () => clearInterval(pollTimer);
    }, [isTraining]);

    // --- EFFECT H: RENDER BOTTOM PANEL TIMELINE CHART ---
    useEffect(() => {
        if (selectedSpotId !== null && mainChartRef.current) {
            const fetchHistory = async () => {
                try {
                    const res = await fetch(`${API_URL}/history?spot_id=${selectedSpotId}`);
                    if (res.ok) {
                        const data = await res.json();
                        
                        const labels = data.history.map(item => {
                            const dateObj = new Date(item.timestamp);
                            return dateObj.toLocaleDateString('en-IN', { weekday: 'short', hour: '2-digit' });
                        });
                        const actuals = data.history.map(item => item.actual_congestion);
                        const predictions = data.history.map(item => item.predicted_congestion);

                        // Clear existing
                        if (mainChartInstance.current) {
                            mainChartInstance.current.destroy();
                        }

                        const ctx = mainChartRef.current.getContext('2d');
                        
                        // Theme colors for labels
                        const tickColor = theme === "dark" ? "#8ea0b4" : "#64748b";
                        const gridColor = theme === "dark" ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";

                        mainChartInstance.current = new Chart(ctx, {
                            type: 'line',
                            data: {
                                labels: labels,
                                datasets: [
                                    {
                                        label: 'Actual Congestion (%)',
                                        data: actuals,
                                        borderColor: '#00f2fe',
                                        backgroundColor: 'rgba(0, 242, 254, 0.03)',
                                        borderWidth: 2,
                                        fill: true,
                                        tension: 0.3,
                                        pointRadius: 1
                                    },
                                    {
                                        label: 'AI Forecasted Risk (%)',
                                        data: predictions,
                                        borderColor: '#00ff87',
                                        borderDash: [5, 5],
                                        backgroundColor: 'transparent',
                                        borderWidth: 2,
                                        fill: false,
                                        tension: 0.3,
                                        pointRadius: 1
                                    }
                                ]
                            },
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                scales: {
                                    x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9, family: "'Share Tech Mono', monospace" } } },
                                    y: { min: 0, max: 100, grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9, family: "'Share Tech Mono', monospace" } } }
                                },
                                plugins: {
                                    legend: { labels: { color: theme === "dark" ? "#f0f4f8" : "#1e293b", font: { size: 10 } } }
                                }
                            }
                        });
                    }
                } catch (err) {
                    console.error("Error rendering details history chart:", err);
                }
            };
            fetchHistory();
        }
    }, [selectedSpotId, theme]);

    // --- EFFECT I: RENDER RIGHT SIDEBAR CHARTS (DAILY PROFILE & VEHICLE DOUGHNUT) ---
    useEffect(() => {
        if (predictions.length > 0) {
            // --- 1. HOURLY DAILY PROFILE BAR CHART ---
            if (hourlyChartRef.current) {
                // Group predictions to get overall hour average risk profile for the date
                const filtered = predictions.filter(p => selectedJurisdiction === "ALL" || p.police_station === selectedJurisdiction);
                const avgRisk = filtered.length > 0 ? (filtered.reduce((sum, item) => sum + item.predicted_congestion, 0) / filtered.length) : 0;
                
                // Let's draw a mock 12-hour projection trend centered around the current hour
                const labels = [];
                const data = [];
                for (let h = -5; h <= 6; h++) {
                    const targetHour = (activeHour + h + 24) % 24;
                    // Apply a cosine curve representing daily rush hour peaks
                    const hourFactor = Math.cos((targetHour - 10) * Math.PI / 12); // peak at 10 AM
                    const simulatedRisk = Math.max(5.0, Math.min(95.0, avgRisk + (hourFactor * 15.0)));
                    
                    labels.push(`${targetHour === 12 ? 12 : targetHour % 12}${targetHour >= 12 ? 'PM' : 'AM'}`);
                    data.push(simulatedRisk);
                }

                if (hourlyChartInstance.current) {
                    hourlyChartInstance.current.destroy();
                }

                const ctx = hourlyChartRef.current.getContext('2d');
                hourlyChartInstance.current = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Avg Risk (%)',
                            data: data,
                            backgroundColor: data.map((val, idx) => idx === 5 ? 'rgba(0, 242, 254, 0.75)' : 'rgba(0, 242, 254, 0.25)'),
                            borderColor: data.map((val, idx) => idx === 5 ? '#00f2fe' : 'rgba(0, 242, 254, 0.45)'),
                            borderWidth: 1,
                            borderRadius: 4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { grid: { display: false }, ticks: { color: theme === "dark" ? "#8ea0b4" : "#64748b", font: { size: 8 } } },
                            y: { min: 0, max: 100, grid: { color: theme === "dark" ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)" }, ticks: { color: theme === "dark" ? "#8ea0b4" : "#64748b", font: { size: 8 } } }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            // --- 2. DYNAMIC VEHICLE TYPE BREAKDOWN DOUGHNUT ---
            if (vehicleChartRef.current) {
                // Simulate vehicle type breakdown (heavy vehicles are higher in early morning, cars during morning rush)
                let heavyPct = 10, carPct = 50, scooterPct = 30, autoPct = 10;
                if (activeHour >= 22 || activeHour <= 5) {
                    heavyPct = 40; carPct = 30; scooterPct = 15; autoPct = 15;
                } else if (activeHour >= 9 && activeHour <= 12) {
                    heavyPct = 5; carPct = 55; scooterPct = 30; autoPct = 10;
                }

                if (vehicleChartInstance.current) {
                    vehicleChartInstance.current.destroy();
                }

                const ctx = vehicleChartRef.current.getContext('2d');
                vehicleChartInstance.current = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Heavy/Commercial', 'Cars/SUVs', 'Scooters/2W', 'Autos/3W'],
                        datasets: [{
                            data: [heavyPct, carPct, scooterPct, autoPct],
                            backgroundColor: ['#ff3838', '#00f2fe', '#00ff87', '#f1c40f'],
                            borderWidth: theme === "dark" ? 2 : 1,
                            borderColor: theme === "dark" ? "#060913" : "#ffffff"
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'right',
                                labels: {
                                    color: theme === "dark" ? "#8ea0b4" : "#64748b",
                                    font: { size: 9 },
                                    boxWidth: 8
                                }
                            }
                        },
                        cutout: '65%'
                    }
                });
            }
        }
    }, [predictions, activeHour, selectedJurisdiction, theme]);

    // Handle retraining trigger
    const triggerRetraining = async () => {
        setIsTraining(true);
        try {
            const res = await fetch(`${API_URL}/retrain`, { method: "POST" });
            if (!res.ok) {
                setIsTraining(false);
                alert("Failed to trigger retraining.");
            }
        } catch (err) {
            setIsTraining(false);
            console.error(err);
        }
    };

    // Filtered spots for Priority List Queue
    const filteredPredictions = predictions.filter(p => 
        selectedJurisdiction === "ALL" || p.police_station === selectedJurisdiction
    );
    // Sort descending by score
    const sortedQueue = [...filteredPredictions].sort((a, b) => b.predicted_congestion - a.predicted_congestion);

    // Get selected spot details
    const selectedSpot = selectedSpotId !== null ? spots[selectedSpotId] : null;
    const selectedRecord = selectedSpotId !== null ? predictions.find(p => p.spot_id === selectedSpotId) : null;

    return (
        <div className="dashboard-container">
            {/* 1. TOP HEADER BAR */}
            <header className="tactical-header">
                <div className="header-logo">
                    <div className="glow-dot"></div>
                    <h1>TRAFFIC HOTSPOTS PREDICTIVE COMMAND CENTER</h1>
                    <span className="badge-sub">BTP AI-INTELLIGENCE v2.0</span>
                </div>
                
                <div className="header-controls">
                    <button 
                        className="btn-theme-toggle" 
                        onClick={() => setTheme(prev => prev === "dark" ? "light" : "dark")}
                    >
                        {theme === "dark" ? "☀️ LIGHT MODE" : "🌙 DARK MODE"}
                    </button>
                    <button 
                        className={`btn-retrain ${isTraining ? 'training' : ''}`}
                        onClick={triggerRetraining}
                        disabled={isTraining}
                    >
                        {isTraining ? "⚡ RETRAINING AI..." : "🔄 RETRAIN MODEL"}
                    </button>
                </div>
            </header>

            {/* 2. THREE-COLUMN DASHBOARD GRID */}
            <main className="dashboard-grid">
                
                {/* COLUMN 1: CONTROLS & PRIORITY LIST */}
                <section className="sidebar-left">
                    <div className="panel-header">
                        <h2>QUEUE CONTROLS</h2>
                    </div>
                    
                    {/* Prediction DateTime Inputs */}
                    <div className="input-control-panel">
                        <div className="input-row">
                            <div className="input-item">
                                <label>PREDICTION DATE</label>
                                <input 
                                    type="date" 
                                    className="input-field"
                                    value={activeDate}
                                    onChange={(e) => setActiveDate(e.target.value)}
                                    min="2024-03-17"
                                    max="2024-04-08"
                                />
                            </div>
                            <div className="input-item">
                                <label>TARGET HOUR</label>
                                <select 
                                    className="input-field"
                                    value={activeHour}
                                    onChange={(e) => setActiveHour(parseInt(e.target.value))}
                                >
                                    {Array.from({ length: 24 }).map((_, h) => (
                                        <option key={h} value={h}>
                                            {h === 0 ? "12:00 AM" : h === 12 ? "12:00 PM" : h > 12 ? `${h - 12}:00 PM` : `${h}:00 AM`}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>
                        <div className="input-item">
                            <label>STATION JURISDICTION</label>
                            <select 
                                className="input-field"
                                value={selectedJurisdiction}
                                onChange={(e) => setSelectedJurisdiction(e.target.value)}
                            >
                                <option value="ALL">ALL JURISDICTIONS</option>
                                <option value="Madiwala">MADIWALA</option>
                                <option value="Bellandur">BELLANDUR</option>
                                <option value="Upparpet">UPPARPET</option>
                                <option value="Shivajinagar">SHIVAJINAGAR</option>
                                <option value="HSR Layout">HSR LAYOUT</option>
                                <option value="K.R. Pura">K.R. PURA</option>
                                <option value="Peenya">PEENYA</option>
                                <option value="Electronic City">ELECTRONIC CITY</option>
                            </select>
                        </div>
                    </div>

                    <div className="panel-header">
                        <h2>PRIORITY HOTSPOTS QUEUE</h2>
                        <span className="pulse-badge">LIVE FORECAST</span>
                    </div>
                    <div className="panel-sub-desc">
                        Ranked by predicted Congestion Risk Index
                    </div>

                    {/* Ranked priority cards scroll area */}
                    <div className="priority-list">
                        {isLoading ? (
                            <div className="loading-spinner">Loading queue...</div>
                        ) : insufficientDataError ? (
                            <div style={{ textAlign: 'center', padding: '20px 15px', color: '#ff3838', fontSize: '0.8rem', lineHeight: '1.4', border: '1px dashed rgba(255,56,56,0.3)', borderRadius: '6px', margin: '10px 0' }}>
                                ⚠️ <strong>INSUFFICIENT DATA</strong><br/>
                                <span style={{ color: 'var(--text-secondary, #8ea0b4)', fontSize: '0.75rem', marginTop: '6px', display: 'block' }}>
                                    {insufficientDataError}
                                </span>
                            </div>
                        ) : sortedQueue.length === 0 ? (
                            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                                No critical hotspots in jurisdiction.
                            </div>
                        ) : (
                            sortedQueue.slice(0, 20).map((spot, idx) => {
                                const score = spot.predicted_congestion;
                                let riskClass = "risk-low";
                                let riskText = "LOW";
                                if (score >= 75) { riskClass = "risk-critical"; riskText = "CRITICAL"; }
                                else if (score >= 50) { riskClass = "risk-high"; riskText = "HIGH"; }
                                else if (score >= 25) { riskClass = "risk-medium"; riskText = "MEDIUM"; }

                                return (
                                    <div 
                                        key={spot.spot_id} 
                                        className={`queue-card ${riskClass} ${selectedSpotId === spot.spot_id ? 'active' : ''}`}
                                        onClick={() => {
                                            setSelectedSpotId(spot.spot_id);
                                            if (mapRef.current) {
                                                mapRef.current.flyTo([spot.latitude, spot.longitude], 14, { duration: 1.2 });
                                            }
                                        }}
                                    >
                                        <div className="card-top">
                                            <span className="card-title">#{idx + 1} - {spot.region_name}</span>
                                            <span className="risk-badge">{riskText}</span>
                                        </div>
                                        <div className="card-details">
                                            <span className="card-station">📍 {spot.police_station}</span>
                                            <div className="card-metrics">
                                                <span>Risk: <strong className="metric-val">{score.toFixed(1)}%</strong></span>
                                                <span>Violations: <strong className="metric-val">{spot.violations_count}</strong></span>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </section>

                {/* COLUMN 2: CENTER MAP */}
                <section className="map-center-panel" style={{ position: 'relative' }}>
                    <div ref={mapContainerRef} className="map-wrapper"></div>
                    {insufficientDataError && (
                        <div className="map-warning-overlay" style={{
                            position: 'absolute',
                            top: '20px',
                            left: '50%',
                            transform: 'translateX(-50%)',
                            zIndex: 1000,
                            background: 'rgba(6, 9, 19, 0.9)',
                            backdropFilter: 'blur(8px)',
                            border: '1px solid rgba(255, 56, 56, 0.4)',
                            padding: '10px 20px',
                            borderRadius: '8px',
                            color: '#fff',
                            fontFamily: "'Outfit', sans-serif",
                            fontSize: '0.8rem',
                            textAlign: 'center',
                            boxShadow: '0 4px 15px rgba(0,0,0,0.5)',
                            pointerEvents: 'none',
                            maxWidth: '80%'
                        }}>
                            <strong>⚠️ MAP CLEAR: INSUFFICIENT DATA</strong><br/>
                            <span style={{ color: '#8ea0b4', fontSize: '0.75rem' }}>No predictions generated for {activeDate}. Select a date between 17 March and 08 April 2024.</span>
                        </div>
                    )}
                    
                    {/* Time Dial Indicator Overlay (floating at bottom of map) */}
                    <div className="time-dial-overlay">
                        <div className="dial-desc">
                            <span className="icon">⚙️</span>
                            <div>
                                <h3>TIME-LAPSE PLAYBACK</h3>
                            </div>
                        </div>
                        <div className="time-dial-controls">
                            <span className="dial-time-display">
                                {activeHour === 0 ? "12:00 AM" : activeHour === 12 ? "12:00 PM" : activeHour > 12 ? `${activeHour - 12}:00 PM` : `${activeHour}:00 AM`}
                            </span>
                            <button 
                                className="btn-tactical" 
                                onClick={() => setIsPlaying(prev => !prev)}
                            >
                                {isPlaying ? "PAUSE" : "PLAY LOOP"}
                            </button>
                        </div>
                    </div>
                </section>

                {/* COLUMN 3: RIGHT PANEL STATS & CHARTS */}
                <section className="sidebar-right">
                    <div className="panel-header">
                        <h2>LIVE KPI METRICS</h2>
                    </div>
                    
                    {/* Live KPIs */}
                    <div className="kpi-container">
                        <div className="kpi-card">
                            <div className="kpi-details">
                                <span className="kpi-title">Average Risk Index</span>
                                <span className="kpi-value">{stats.average_congestion_risk ? stats.average_congestion_risk.toFixed(1) : 0}%</span>
                            </div>
                            <span className="kpi-icon">📈</span>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-details">
                                <span className="kpi-title">Critical Choke Zones</span>
                                <span className="kpi-value critical">{stats.active_critical_hotspots}</span>
                            </div>
                            <span className="kpi-icon">🚨</span>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-details">
                                <span className="kpi-title">Total Active Violations</span>
                                <span className="kpi-value">{stats.total_day_violations}</span>
                            </div>
                            <span className="kpi-icon">🚗</span>
                        </div>
                    </div>

                    <div className="panel-header">
                        <h2>CONGESTION ANALYTICS</h2>
                    </div>
                    
                    {/* Analytics charts lists */}
                    <div className="sidebar-charts-container">
                        <div className="chart-box">
                            <span className="chart-box-title">Projected Hourly Risk profile</span>
                            <div className="small-chart-container">
                                <canvas ref={hourlyChartRef}></canvas>
                            </div>
                        </div>
                        <div className="chart-box">
                            <span className="chart-box-title">Traffic Obstruction Sources</span>
                            <div className="small-chart-container">
                                <canvas ref={vehicleChartRef}></canvas>
                            </div>
                        </div>
                    </div>
                </section>

                {/* 3. BOTTOM SLIDE-UP HISTORICAL COMPARISON PANEL */}
                <footer className={`bottom-analytics-overlay ${selectedSpotId === null ? 'hidden' : ''}`}>
                    <button className="btn-close-overlay" onClick={() => setSelectedSpotId(null)}>&times;</button>
                    <div className="overlay-content">
                        {selectedSpotId !== null && selectedSpot && selectedRecord ? (
                            <div className="overlay-details-panel">
                                <div>
                                    <h3>{selectedSpot.region_name}</h3>
                                    <div className="overlay-meta-grid">
                                        <div className="overlay-meta-row">
                                            <span className="overlay-meta-label">Station Jurisdiction:</span>
                                            <span className="overlay-meta-val">{selectedSpot.police_station.toUpperCase()}</span>
                                        </div>
                                        <div className="overlay-meta-row">
                                            <span className="overlay-meta-label">Coordinates:</span>
                                            <span className="overlay-meta-val">{selectedSpot.latitude.toFixed(5)}, {selectedSpot.longitude.toFixed(5)}</span>
                                        </div>
                                    </div>
                                </div>
                                
                                <div className="overlay-score-row">
                                    <div className="overlay-score-card">
                                        <span className="overlay-card-lbl">Congestion Risk</span>
                                        <span className={`overlay-card-val critical`}>
                                            {selectedRecord.predicted_congestion.toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="overlay-score-card">
                                        <span className="overlay-card-lbl">Baseline Avg</span>
                                        <span className="overlay-card-val">
                                            {selectedSpot.baseline_congestion ? selectedSpot.baseline_congestion.toFixed(1) : "0.0"}%
                                        </span>
                                    </div>
                                    <div className="overlay-score-card">
                                        <span className="overlay-card-lbl">Hourly Violations</span>
                                        <span className="overlay-card-val">{selectedRecord.violations_count}</span>
                                    </div>
                                </div>

                                <div className="overlay-recommendation-box">
                                    <span className="overlay-rec-icon">⚠️</span>
                                    <div className="overlay-rec-text">
                                        <strong>PATROL ROUTING RECOMMENDATION:</strong>
                                        <p>
                                            {selectedRecord.predicted_congestion >= 75 ? (
                                                `🚨 CRITICAL. Deploy 1 towing truck and dispatch 2 officers to clear main road crossing footprint immediately.`
                                            ) : selectedRecord.predicted_congestion >= 50 ? (
                                                `⚠️ HIGH RISK. Dispatch 1 patrol officer to issue parking tickets and warning notices.`
                                            ) : selectedRecord.predicted_congestion >= 25 ? (
                                                `⚡ MODERATE. Include region in standard hourly patrol route check.`
                                            ) : (
                                                `✅ LOW. Flow normal. Surveillance cameras are sufficient.`
                                            )}
                                        </p>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            <div style={{ color: 'var(--text-secondary)', padding: '20px' }}>Loading spot info...</div>
                        )}

                        <div className="overlay-chart-wrapper">
                            <canvas ref={mainChartRef}></canvas>
                        </div>
                    </div>
                </footer>
            </main>
        </div>
    );
}
