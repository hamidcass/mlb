import { useState, useMemo, useEffect } from "react";
import {
    ScatterChart,
    Scatter,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    BarChart,
    Bar,
    Cell,
} from "recharts";
import { fetchAllPredictions, fetchMetrics, fetchImportance } from "../api/api";

interface Prediction {
    Player: string;
    Actual: number;
    Predicted: number;
    Error: number;
    Abs_Error: number;
    Pct_Error: number;
    Age?: number;
    PA?: number;
}

interface Metrics {
    MAE: number;
    R2: number;
    Num_Players: number;
}

interface FeatureImportance {
    Feature: string;
    Importance: number;
    Direction?: number;
    Effect?: string;
}

interface CustomTooltipProps {
    active?: boolean;
    payload?: Array<{ payload: Prediction }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
    if (active && payload && payload.length) {
        const data = payload[0].payload;
        return (
            <div className="scatter-tooltip">
                <p className="tooltip-name">{data.Player}</p>
                <p>Actual: <strong>{data.Actual.toFixed(3)}</strong></p>
                <p>Predicted: <strong>{data.Predicted.toFixed(3)}</strong></p>
                <p>Error: <span className={data.Error > 0 ? "positive" : "negative"}>
                    {data.Error > 0 ? "+" : ""}{data.Error.toFixed(3)}
                </span></p>
            </div>
        );
    }
    return null;
}

function LoadingSkeleton() {
    return (
        <>
            {/* Metrics Skeleton */}
            <section className="skeleton-section">
                <div className="skeleton skeleton-title"></div>
                <div className="skeleton-metrics-grid">
                    <div className="skeleton skeleton-metric-card"></div>
                    <div className="skeleton skeleton-metric-card"></div>
                    <div className="skeleton skeleton-metric-card"></div>
                </div>
            </section>

            {/* Feature Importance Skeleton */}
            <section className="skeleton-section">
                <div className="skeleton skeleton-title"></div>
                <div className="skeleton skeleton-subtitle"></div>
                <div className="skeleton skeleton-importance"></div>
            </section>

            {/* Chart Skeleton */}
            <section className="skeleton-section">
                <div className="skeleton skeleton-title"></div>
                <div className="skeleton skeleton-subtitle"></div>
                <div className="skeleton skeleton-chart"></div>
            </section>

            {/* Tables Skeleton */}
            <section className="skeleton-section">
                <div className="skeleton-performers-grid">
                    <div className="skeleton skeleton-table"></div>
                    <div className="skeleton skeleton-table"></div>
                </div>
            </section>
        </>
    );
}

export default function Predictions() {
    const [targetStat, setTargetStat] = useState("OPS");
    const [model, setModel] = useState("XGBoost");
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    const [metrics, setMetrics] = useState<Metrics | null>(null);
    const [importance, setImportance] = useState<FeatureImportance[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Zoom state (10 = no zoom, 100 = max zoom)
    const [zoomLevel, setZoomLevel] = useState(10);
    const [panX, setPanX] = useState(50);
    const [panY, setPanY] = useState(50);

    // Player search state
    const [searchQuery, setSearchQuery] = useState("");
    const [highlightedPlayer, setHighlightedPlayer] = useState<string | null>(null);

    // Auto-load predictions on mount with default settings (XGBoost + OPS)
    const loadPredictions = async (stat: string, modelName: string) => {
        setLoading(true);
        setError(null);
        setZoomLevel(10);
        setPanX(50);
        setPanY(50);
        setSearchQuery("");
        setHighlightedPlayer(null);
        try {
            const [predData, metricsData] = await Promise.all([
                fetchAllPredictions(stat, modelName),
                fetchMetrics(stat, modelName)
            ]);
            setPredictions(predData.predictions || []);
            setMetrics(metricsData);

            // Try to fetch importance (may fail for some models)
            try {
                const importanceData = await fetchImportance(stat, modelName);
                setImportance(importanceData.features || []);
            } catch {
                setImportance([]);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to fetch data");
            setPredictions([]);
            setMetrics(null);
            setImportance([]);
        } finally {
            setLoading(false);
        }
    };

    // Load default predictions on page mount
    useEffect(() => {
        loadPredictions(targetStat, model);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);


    const handleRunPredictions = () => {
        loadPredictions(targetStat, model);
    };

    // Calculate base domain
    const allValues = predictions.flatMap(p => [p.Actual, p.Predicted]);
    const dataMin = allValues.length ? Math.min(...allValues) : 0;
    const dataMax = allValues.length ? Math.max(...allValues) : 1;
    const dataRange = dataMax - dataMin;
    const baseMin = dataMin - dataRange * 0.05;
    const baseMax = dataMax + dataRange * 0.05;
    const fullRange = baseMax - baseMin;

    // Calculate zoomed domain (invert zoomLevel so 100 = full view, lower = more zoom)
    const zoomedDomain = useMemo(() => {
        const effectiveZoom = 110 - zoomLevel; // Invert: slider 10 = view 100%, slider 100 = view 10%
        const viewRange = (fullRange * effectiveZoom) / 100;
        const halfView = viewRange / 2;
        const centerX = baseMin + (fullRange * panX) / 100;
        const centerY = baseMin + (fullRange * panY) / 100;
        return {
            xMin: Math.max(baseMin, centerX - halfView),
            xMax: Math.min(baseMax, centerX + halfView),
            yMin: Math.max(baseMin, centerY - halfView),
            yMax: Math.min(baseMax, centerY + halfView),
        };
    }, [zoomLevel, panX, panY, baseMin, baseMax, fullRange]);

    // Filter predictions for search
    const filteredPlayers = useMemo(() => {
        if (!searchQuery.trim()) return [];
        return predictions
            .filter(p => p.Player.toLowerCase().includes(searchQuery.toLowerCase()))
            .slice(0, 8);
    }, [predictions, searchQuery]);

    const handleResetZoom = () => {
        setZoomLevel(10);
        setPanX(50);
        setPanY(50);
    };

    const handlePlayerSelect = (playerName: string) => {
        setHighlightedPlayer(playerName);
        setSearchQuery("");
    };

    // Prepare importance data for chart (top 10)
    const importanceChartData = importance.slice(0, 10).map(f => ({
        ...f,
        Feature: f.Feature.replace("Current_", ""),
        fill: f.Direction && f.Direction > 0 ? "#4dc9ff" : "#ff6b5b"
    }));

    return (
        <div className="page-container">
            {/* Page Header */}
            <header className="hero-header">
                <h1>Run Predictions</h1>
                <p className="subtitle">
                    Select your target stat and model to generate{" "}
                    <span className="highlight">2025 MLB projections</span>.
                </p>
            </header>

            {/* Control Panel */}
            <div className="control-panel">
                <div className="control-group">
                    <label>Target Stat</label>
                    <select
                        value={targetStat}
                        onChange={(e) => setTargetStat(e.target.value)}
                    >
                        <option value="OPS">OPS</option>
                        <option value="AVG">AVG</option>
                        <option value="HR">HR</option>
                        <option value="wRC_PLUS">wRC+</option>
                    </select>
                </div>

                <div className="control-group">
                    <label>Model</label>
                    <select value={model} onChange={(e) => setModel(e.target.value)}>
                        <option value="XGBoost">XGBoost</option>
                        <option value="RandomForest">Random Forest</option>
                        <option value="LinearRegression">Linear Regression</option>
                        <option value="Ridge">Ridge</option>
                    </select>
                </div>

                <button
                    className="btn-primary"
                    onClick={handleRunPredictions}
                    disabled={loading}
                >
                    {loading ? "Loading..." : "Run Predictions"}
                </button>
            </div>

            {/* Error Message */}
            {error && (
                <div className="error-banner">
                    {error}
                </div>
            )}

            {/* Loading Skeleton */}
            {loading && <LoadingSkeleton />}

            {/* Model Metrics */}
            {!loading && metrics && (
                <section className="metrics-section">
                    <h2>Model Performance</h2>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <div className="metric-label">Mean Absolute Error</div>
                            <div className="metric-value">{metrics.MAE.toFixed(4)}</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">R² Score</div>
                            <div className="metric-value">{metrics.R2.toFixed(4)}</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">Players Evaluated</div>
                            <div className="metric-value">{metrics.Num_Players}</div>
                        </div>
                    </div>
                </section>
            )}

            {/* Feature Importance Chart */}
            {!loading && importance.length > 0 && (
                <section className="importance-section">
                    <h2>Feature Importance</h2>
                    <p className="chart-subtitle">
                        Top factors driving {model} predictions for {targetStat}
                    </p>
                    <div className="importance-chart-container">
                        <div className="chart-scroll-wrapper">
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart
                                    data={importanceChartData}
                                    layout="vertical"
                                    margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                                    <XAxis type="number" tick={{ fill: "#8b949e" }} />
                                    <YAxis
                                        type="category"
                                        dataKey="Feature"
                                        tick={{ fill: "#8b949e", fontSize: 12 }}
                                        width={90}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            background: "#1a1f2e",
                                            border: "1px solid #2d3748",
                                            borderRadius: "8px",
                                        }}
                                        labelStyle={{ color: "#e6edf3" }}
                                        itemStyle={{ color: "#e6edf3" }}
                                    />
                                    <Bar dataKey="Importance" radius={[0, 4, 4, 0]}>
                                        {importanceChartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.fill} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <p className="importance-legend">
                        <span className="legend-dot cyan"></span> Increases prediction
                        <span className="legend-dot coral"></span> Decreases prediction
                    </p>
                </section>
            )}

            {/* Scatter Plot */}
            {!loading && predictions.length > 0 && (
                <section className="chart-section">
                    <h2>Predicted vs Actual {targetStat}</h2>
                    <p className="chart-subtitle">
                        {predictions.length} players • {model} model
                        {highlightedPlayer && <span className="highlight-badge"> • Showing: {highlightedPlayer}</span>}
                    </p>

                    {/* Player Search */}
                    <div className="player-search">
                        <input
                            type="text"
                            placeholder="Search for a player..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="search-input"
                        />
                        {filteredPlayers.length > 0 && (
                            <div className="search-results">
                                {filteredPlayers.map((p, i) => (
                                    <div
                                        key={i}
                                        className="search-result-item"
                                        onClick={() => handlePlayerSelect(p.Player)}
                                    >
                                        <span className="player-name">{p.Player}</span>
                                        <span className="player-stats">
                                            Actual: {p.Actual.toFixed(3)} | Pred: {p.Predicted.toFixed(3)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        )}
                        {highlightedPlayer && (
                            <button
                                className="btn-clear-highlight"
                                onClick={() => setHighlightedPlayer(null)}
                            >
                                Clear highlight
                            </button>
                        )}
                    </div>

                    {/* Zoom Controls */}
                    <div className="zoom-controls">
                        <div className="zoom-control-group">
                            <label>Zoom</label>
                            <input
                                type="range"
                                min="10"
                                max="100"
                                value={zoomLevel}
                                onChange={(e) => setZoomLevel(Number(e.target.value))}
                                className="zoom-slider"
                            />
                            <span className="zoom-value">{Math.round(zoomLevel / 10)}x</span>
                        </div>
                        <div className="zoom-control-group">
                            <label>Pan X</label>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                value={panX}
                                onChange={(e) => setPanX(Number(e.target.value))}
                                className="zoom-slider"
                                disabled={zoomLevel === 10}
                            />
                        </div>
                        <div className="zoom-control-group">
                            <label>Pan Y</label>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                value={panY}
                                onChange={(e) => setPanY(Number(e.target.value))}
                                className="zoom-slider"
                                disabled={zoomLevel === 10}
                            />
                        </div>
                        <button className="btn-reset" onClick={handleResetZoom}>
                            Reset View
                        </button>
                    </div>

                    <div className="scatter-container">
                        <div className="chart-scroll-wrapper">
                            <ResponsiveContainer width="100%" height={500}>
                                <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 60 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                                    <XAxis
                                        type="number"
                                        dataKey="Actual"
                                        name="Actual"
                                        domain={[zoomedDomain.xMin, zoomedDomain.xMax]}
                                        tick={{ fill: "#8b949e" }}
                                        label={{
                                            value: `Actual ${targetStat}`,
                                            position: "bottom",
                                            offset: 40,
                                            fill: "#8b949e",
                                        }}
                                        allowDataOverflow
                                    />
                                    <YAxis
                                        type="number"
                                        dataKey="Predicted"
                                        name="Predicted"
                                        domain={[zoomedDomain.yMin, zoomedDomain.yMax]}
                                        tick={{ fill: "#8b949e" }}
                                        label={{
                                            value: `Predicted ${targetStat}`,
                                            angle: -90,
                                            position: "left",
                                            offset: 40,
                                            fill: "#8b949e",
                                        }}
                                        allowDataOverflow
                                    />
                                    <Tooltip content={<CustomTooltip />} />
                                    <ReferenceLine
                                        segment={[
                                            { x: zoomedDomain.xMin, y: zoomedDomain.xMin },
                                            { x: zoomedDomain.xMax, y: zoomedDomain.xMax },
                                        ]}
                                        stroke="#ff6b5b"
                                        strokeDasharray="5 5"
                                        strokeWidth={2}
                                    />
                                    {/* Regular scatter points */}
                                    <Scatter
                                        data={predictions.filter(p => p.Player !== highlightedPlayer)}
                                        fill="#4dc9ff"
                                        fillOpacity={highlightedPlayer ? 0.3 : 0.7}
                                    />
                                    {/* Highlighted player */}
                                    {highlightedPlayer && (
                                        <Scatter
                                            data={predictions.filter(p => p.Player === highlightedPlayer)}
                                            fill="#fbbf24"
                                            fillOpacity={1}
                                            shape="star"
                                        />
                                    )}
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <p className="chart-legend">
                        <span className="legend-line"></span> Perfect prediction line (y = x)
                    </p>
                </section>
            )}

            {/* Top Performers Tables */}
            {!loading && predictions.length > 0 && (
                <section className="performers-section">
                    <div className="performers-grid">
                        {/* Overperformers */}
                        <div className="performers-table">
                            <h3>Top 5 Overperformers</h3>
                            <div className="table-scroll-wrapper">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Player</th>
                                            <th>Predicted</th>
                                            <th>Actual</th>
                                            <th>Error</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {[...predictions]
                                            .sort((a, b) => a.Error - b.Error)
                                            .slice(0, 5)
                                            .map((p, i) => (
                                                <tr key={i}>
                                                    <td className="player-name">{p.Player}</td>
                                                    <td>{p.Predicted.toFixed(3)}</td>
                                                    <td>{p.Actual.toFixed(3)}</td>
                                                    <td className="error-cell overperform">{p.Error.toFixed(3)}</td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Underperformers */}
                        <div className="performers-table">
                            <h3>Top 5 Underperformers</h3>
                            <div className="table-scroll-wrapper">
                                <table>
                                    <thead>
                                        <tr>
                                            <th>Player</th>
                                            <th>Predicted</th>
                                            <th>Actual</th>
                                            <th>Error</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {[...predictions]
                                            .sort((a, b) => b.Error - a.Error)
                                            .slice(0, 5)
                                            .map((p, i) => (
                                                <tr key={i}>
                                                    <td className="player-name">{p.Player}</td>
                                                    <td>{p.Predicted.toFixed(3)}</td>
                                                    <td>{p.Actual.toFixed(3)}</td>
                                                    <td className="error-cell underperform">+{p.Error.toFixed(3)}</td>
                                                </tr>
                                            ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </section>
            )}

            {/* Info Banner */}
            <div className="info-banner">
                Predict 2025 player stats using 9 years of historical data, advanced sabermetrics,
                and various ML models. Configure your settings and click Run Predictions to get started.
            </div>
        </div>
    );
}
