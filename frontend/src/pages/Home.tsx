import { Link } from "react-router-dom";
import { useState, useEffect } from "react";
import { fetchStats } from "../api/api";

interface DatasetStats {
  total_player_seasons: number;
  unique_players: number;
  years: number[];
}

export default function Home() {
  const [stats, setStats] = useState<DatasetStats | null>(null);

  useEffect(() => {
    fetchStats()
      .then(data => setStats(data))
      .catch(() => {
        // Fallback to defaults
        setStats({
          total_player_seasons: 1709,
          unique_players: 462,
          years: [2020, 2021, 2022, 2023, 2024, 2025]
        });
      });
  }, []);

  const formatYears = (years: number[]) => {
    if (years.length < 2) return years.join(", ");
    // Training data years (exclude 2025 which is prediction year)
    const trainingYears = years.filter(y => y < 2025);
    if (trainingYears.length < 2) return trainingYears.join(", ");
    return `${trainingYears[0]}–${trainingYears[trainingYears.length - 1]}`;
  };

  return (
    <div className="page-container">
      {/* Hero Header */}
      <header className="hero-header hero-landing">
        <h1>2025 MLB Offensive Projections</h1>
        <p className="subtitle">
          Harness the power of <span className="highlight">machine learning</span> to predict
          player performance with advanced metrics and multiple ML models.
        </p>
        <p className="hero-description">
          Our platform combines 9 years of historical data, advanced sabermetrics,
          and state-of-the-art machine learning algorithms to generate accurate
          2025 season projections for every MLB player.
        </p>
        <Link to="/predictions" className="btn-cta">
          Analyze Predictions
        </Link>
      </header>

      {/* Dataset Overview */}
      <section className="stats-section">
        <h2>Dataset Overview</h2>
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-label">Total Player-Seasons</div>
            <div className="stat-value">
              {stats ? stats.total_player_seasons.toLocaleString() : "—"}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Years Covered</div>
            <div className="stat-value">
              {stats ? formatYears(stats.years) : "—"}
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Models Available</div>
            <div className="stat-value">4</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Stats Predicted</div>
            <div className="stat-value">4</div>
          </div>
        </div>
      </section>
    </div>
  );
}
