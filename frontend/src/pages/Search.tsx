import { useState, useEffect, useMemo } from "react";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
    ReferenceDot,
} from "recharts";
import { fetchPlayers, fetchPlayer, fetchPlayerHistory } from "../api/api";

interface Player {
    Player: string;
    Team: string;
    Age: number;
    PA: number;
}

interface PlayerPrediction {
    stat: string;
    model: string;
    Predicted: number;
    Actual: number;
    Error: number;
}

interface HistoryPoint {
    Season: number;
    OPS: number;
}

// Calculate prediction grade based on error percentage
function calculateGrade(predicted: number, actual: number): string {
    const pctError = Math.abs((predicted - actual) / actual) * 100;
    if (pctError <= 2) return "A+";
    if (pctError <= 5) return "A";
    if (pctError <= 8) return "B+";
    if (pctError <= 12) return "B";
    if (pctError <= 18) return "C+";
    if (pctError <= 25) return "C";
    return "D";
}

function getGradeColor(grade: string): string {
    if (grade.startsWith("A")) return "#22c55e";
    if (grade.startsWith("B")) return "#4dc9ff";
    if (grade.startsWith("C")) return "#fbbf24";
    return "#f43f5e";
}

// Model options for the dropdown
const MODEL_OPTIONS = [
    { value: "linearregression", label: "Linear Regression" },
    { value: "ridge", label: "Ridge" },
    { value: "randomforest", label: "Random Forest" },
    { value: "xgboost", label: "XGBoost" },
];

export default function Search() {
    const [players, setPlayers] = useState<Player[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedPlayer, setSelectedPlayer] = useState<Player | null>(null);
    const [playerPredictions, setPlayerPredictions] = useState<PlayerPrediction[]>([]);
    const [playerHistory, setPlayerHistory] = useState<HistoryPoint[]>([]);
    const [loading, setLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);
    const [selectedModel, setSelectedModel] = useState("xgboost");

    // Load players list on mount
    useEffect(() => {
        fetchPlayers()
            .then((data) => setPlayers(data.players || []))
            .catch(console.error);
    }, []);

    // Filter players based on search
    const filteredPlayers = useMemo(() => {
        if (!searchQuery.trim()) return [];
        return players
            .filter((p) => p.Player.toLowerCase().includes(searchQuery.toLowerCase()))
            .slice(0, 10);
    }, [players, searchQuery]);

    // Handle player selection
    const handlePlayerSelect = async (player: Player) => {
        setSelectedPlayer(player);
        setSearchQuery(player.Player);
        setShowDropdown(false);
        setLoading(true);

        try {
            // Fetch player predictions across all models/stats
            const predData = await fetchPlayer(player.Player);
            setPlayerPredictions(predData.predictions || []);

            // Fetch historical data (for actual OPS by season)
            const historyData = await fetchPlayerHistory(player.Player);
            setPlayerHistory(historyData.history || []);
        } catch (err) {
            console.error("Error fetching player data:", err);
        } finally {
            setLoading(false);
        }
    };

    // Get OPS prediction based on selected model
    const opsPrediction = playerPredictions.find(
        (p) => p.stat === "OPS" && p.model === selectedModel
    );


    // Get model display name
    const selectedModelLabel = MODEL_OPTIONS.find(m => m.value === selectedModel)?.label || selectedModel;

    // Prepare chart data with prediction point - uses selected model's prediction
    const chartData = useMemo(() => {
        const data: { Season: number; Actual: number | undefined; Predicted: number | null }[] =
            playerHistory.map((h) => ({
                Season: h.Season,
                Actual: h.OPS,
                Predicted: null as number | null,
            }));

        // Add 2025 prediction point from selected model
        const modelPrediction = opsPrediction?.Predicted ?? null;
        if (modelPrediction !== null) {
            const existing2025 = data.find((d) => d.Season === 2025);
            if (existing2025) {
                existing2025.Predicted = modelPrediction;
            } else {
                data.push({
                    Season: 2025,
                    Actual: opsPrediction?.Actual,
                    Predicted: modelPrediction,
                });
            }
        }

        return data.sort((a, b) => a.Season - b.Season);
    }, [playerHistory, opsPrediction]);

    return (
        <div className="page-container search-page">
            {/* Search Header */}
            <div className="search-header">
                <label className="search-label">Search for a player</label>
                <div className="search-dropdown-container">
                    <input
                        type="text"
                        className="search-dropdown-input"
                        placeholder="Vladimir Guerrero Jr."
                        value={searchQuery}
                        onChange={(e) => {
                            setSearchQuery(e.target.value);
                            setShowDropdown(true);
                        }}
                        onFocus={() => setShowDropdown(true)}
                    />
                    <span className="dropdown-arrow">▼</span>
                    {showDropdown && filteredPlayers.length > 0 && (
                        <div className="search-dropdown-results">
                            {filteredPlayers.map((p, i) => (
                                <div
                                    key={i}
                                    className="search-dropdown-item"
                                    onClick={() => handlePlayerSelect(p)}
                                >
                                    <span className="player-name">{p.Player}</span>
                                    <span className="player-team">{p.Team}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Player Content */}
            {selectedPlayer && !loading && (
                <>
                    {/* Player Name */}
                    <h1 className="player-title">
                        {selectedPlayer.Player}
                        {/* {selectedPlayer.Player} <span className="player-position">1B</span> */}
                    </h1>

                    {/* Player Info Section */}
                    <section className="player-info-section">
                        <h2 className="section-title">Player Info</h2>
                        <div className="player-info-grid">
                            <div className="info-item">
                                <span className="info-label">Team:</span>
                                <span className="info-value team-badge">
                                    {selectedPlayer.Team} → {selectedPlayer.Team}
                                </span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Age:</span>
                                <span className="info-value">{selectedPlayer.Age}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">PA:</span>
                                <span className="info-value">{selectedPlayer.PA}</span>
                            </div>
                        </div>
                    </section>

                    {/* Prediction Summary Section */}
                    {opsPrediction && (
                        <section className="prediction-summary-section">
                            <div className="prediction-header">
                                <h2 className="section-title prediction-title">
                                    Prediction Summary{" "}
                                    <span className="model-badge">({selectedModelLabel})</span>
                                </h2>
                                <div className="model-selector">
                                    <label className="model-selector-label">Model:</label>
                                    <select
                                        className="model-select"
                                        value={selectedModel}
                                        onChange={(e) => setSelectedModel(e.target.value)}
                                    >
                                        {MODEL_OPTIONS.map((opt) => (
                                            <option key={opt.value} value={opt.value}>
                                                {opt.label}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                            <div className="prediction-cards">
                                <div className="prediction-card">
                                    <span className="prediction-label">Predicted 2025 OPS</span>
                                    <span className="prediction-value">
                                        {opsPrediction.Predicted.toFixed(3)}
                                    </span>
                                </div>
                                <div className="prediction-card">
                                    <span className="prediction-label">Actual 2025 OPS</span>
                                    <span className="prediction-value">
                                        {opsPrediction.Actual.toFixed(3)}
                                    </span>
                                    <span
                                        className={`prediction-diff ${opsPrediction.Error < 0 ? "positive" : "negative"
                                            }`}
                                    >
                                        {opsPrediction.Error > 0 ? "+" : ""}
                                        {opsPrediction.Error.toFixed(3)}
                                    </span>
                                </div>
                                <div className="prediction-card">
                                    <span className="prediction-label">Prediction Score</span>
                                    <span
                                        className="prediction-grade"
                                        style={{
                                            color: getGradeColor(
                                                calculateGrade(opsPrediction.Predicted, opsPrediction.Actual)
                                            ),
                                        }}
                                    >
                                        {calculateGrade(opsPrediction.Predicted, opsPrediction.Actual)}
                                    </span>
                                </div>
                            </div>
                        </section>
                    )}

                    {/* Historical Performance Chart */}
                    {chartData.length > 0 && (
                        <section className="history-section">
                            <h2 className="section-title">Historical Performance</h2>
                            <div className="history-chart-container">
                                <div className="chart-scroll-wrapper">
                                    <ResponsiveContainer width="100%" height={350}>
                                        <LineChart
                                            data={chartData}
                                            margin={{ top: 20, right: 80, bottom: 20, left: 40 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                                            <XAxis
                                                dataKey="Season"
                                                tick={{ fill: "#8b949e" }}
                                                tickFormatter={(val) => val.toString()}
                                            />
                                            <YAxis
                                                domain={["auto", "auto"]}
                                                tick={{ fill: "#8b949e" }}
                                                tickFormatter={(val) => val.toFixed(2)}
                                                label={{
                                                    value: "OPS",
                                                    angle: -90,
                                                    position: "insideLeft",
                                                    fill: "#8b949e",
                                                }}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    background: "#1a222d",
                                                    border: "1px solid #2d3748",
                                                    borderRadius: "8px",
                                                }}
                                                labelStyle={{ color: "#f0f6fc" }}
                                            />
                                            <Legend
                                                verticalAlign="top"
                                                align="right"
                                                wrapperStyle={{ paddingBottom: 20 }}
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="Actual"
                                                stroke="#4dc9ff"
                                                strokeWidth={2}
                                                dot={{ fill: "#4dc9ff", r: 5 }}
                                                activeDot={{ r: 7 }}
                                                name="Actual"
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="Predicted"
                                                stroke="#ff6b5b"
                                                strokeWidth={2}
                                                strokeDasharray="5 5"
                                                dot={{ fill: "#ff6b5b", r: 5 }}
                                                name="Prediction"
                                                connectNulls={false}
                                            />
                                            {opsPrediction && (
                                                <ReferenceDot
                                                    x={2025}
                                                    y={opsPrediction.Predicted}
                                                    r={6}
                                                    fill="#ff6b5b"
                                                    stroke="#ff6b5b"
                                                />
                                            )}
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </section>
                    )}
                </>
            )}

            {/* Loading State - Skeleton */}
            {loading && (
                <div className="search-loading-skeleton">
                    {/* Player name skeleton */}
                    <div className="skeleton skeleton-player-name"></div>

                    {/* Player Info skeleton */}
                    <section className="skeleton-section">
                        <div className="skeleton skeleton-title" style={{ width: '120px' }}></div>
                        <div className="skeleton-info-grid">
                            <div className="skeleton skeleton-info-item"></div>
                            <div className="skeleton skeleton-info-item"></div>
                            <div className="skeleton skeleton-info-item"></div>
                        </div>
                    </section>

                    {/* Prediction Summary skeleton */}
                    <section className="skeleton-section">
                        <div className="skeleton skeleton-title" style={{ width: '200px' }}></div>
                        <div className="skeleton-prediction-cards">
                            <div className="skeleton skeleton-prediction-card"></div>
                            <div className="skeleton skeleton-prediction-card"></div>
                            <div className="skeleton skeleton-prediction-card"></div>
                        </div>
                    </section>

                    {/* Chart skeleton */}
                    <section className="skeleton-section">
                        <div className="skeleton skeleton-title" style={{ width: '180px' }}></div>
                        <div className="skeleton skeleton-history-chart"></div>
                    </section>
                </div>
            )}

            {/* Empty State */}
            {!selectedPlayer && !loading && (
                <div className="empty-state">
                    <p>Search for a player to view their predictions and performance.</p>
                </div>
            )}
        </div>
    );
}