import { Link } from "react-router-dom";

export default function Home() {
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
          Our platform combines 4 years of historical data, advanced sabermetrics,
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
            <div className="stat-value">1,186</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Years Covered</div>
            <div className="stat-value">2020–2024</div>
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
