from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from io import BytesIO
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="MLB Prediction API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://main.d310rgog3a4st0.amplifyapp.com",
    "https://api.inningai.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Helper function to load batting data with fallbacks
def load_batting_data() -> pd.DataFrame:
    """
    Load batting data from S3 first, then local fallbacks.
    Returns a DataFrame or raises an exception if all sources fail.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    backend_dir = script_dir.parent  # backend/
    project_root = backend_dir.parent  # mlb/
    
    # 1) Try S3 first
    try:
        import boto3
        s3 = boto3.client('s3')
        buffer = BytesIO()
        s3.download_fileobj(Bucket="mlb-ml-data", Key="raw/batting.parquet", Fileobj=buffer)
        buffer.seek(0)
        df = pd.read_parquet(buffer)
        print("load_batting_data: Successfully loaded from S3")
        return df
    except Exception as s3_err:
        print(f"load_batting_data: S3 load failed: {s3_err}")
    
    # 2) Try local fallbacks
    local_paths = [
        backend_dir / "data" / "raw" / "batting.parquet",
        project_root / "backend" / "data" / "raw" / "batting.parquet",
        Path("data/raw/batting.parquet"),
        Path("backend/data/raw/batting.parquet"),
    ]
    
    for path in local_paths:
        try:
            df = pd.read_parquet(path)
            print(f"load_batting_data: Successfully loaded from {path}")
            return df
        except Exception as local_err:
            print(f"load_batting_data: could not load {path}: {local_err}")
    
    raise FileNotFoundError("Could not load batting data from any source")



@app.get("/")
def root():
    return {"message": "MLB Prediction API is running."}

@app.get("/predictions")
def get_predictions(stat: str, model: str, limit: int = 10000):
    
    """
        Retrieve the latest predictions for a specified model
        Ex: /predictions?stat=HR&model=XGBoost
    """

    table_name = f"{stat.lower()}_{model.lower()}_predictions"

    q = text(f"""
        SELECT * 
        FROM {table_name}
        ORDER BY "Player"
        LIMIT :limit
             """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(q, {"limit": limit})
            rows = [dict(row._mapping) for row in result]
        return {
            "stat": stat,
            "model": model,
            "count": len(rows),
            "predictions": rows
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/player/{player_name}")
def get_player_prediction(player_name: str):
    """
        Retrieve all predictions for a specified player across all models
        Ex: /player/Mike Trout
    """

    stats = ["hr", "avg", "ops", "wrc_plus"]
    # stat_changed = stat.lower().replace("+", "plus")
    models = ["linearregression", "ridge", "randomforest", "xgboost"]

    results = []

  
    try:
        with engine.connect() as conn:
            for stat in stats:
                for model in models:
                    table_name = f'"{stat}_{model}_predictions"'
                    q = text(f"""
                        SELECT *, :stat AS stat, :model AS model 
                        FROM {table_name}
                        WHERE "Player" ILIKE :player
                        ORDER BY "Next_Season" DESC
                        LIMIT 1
                             """)
                    result = conn.execute(q, {
                        "player": f"%{player_name}%",
                        "stat": stat.upper(),
                        "model": model
                    })
                    row = result.fetchone()
                    if row:
                        results.append(dict(row._mapping))
        if not results:
            raise HTTPException(status_code=404, detail="Player not found")
        
        return {
            "player": player_name,
            "count": len(results),
            "predictions": results
        }
    

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/meta")
def get_metadata():

    stats = ["hr", "avg", "ops", "wrc_plus"]
    models = ["LinearRegression", "Ridge", "RandomForest", "XGBoost"]

    combos = [
        {"stat": s, "model": m}
        for s in stats
        for m in models
    ]

    return {"available_predictions": combos}

@app.get("/stats")
def get_stats():
    """
        Retrieve dataset statistics for the landing page
        Returns total player-seasons from raw batting data
    """
    # Hardcoded values for 2016-2024 training data (use when S3 unavailable)
    EXPECTED_STATS = {
        "total_player_seasons": 2500,
        "unique_players": 500,
        "years": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    }
    
    # Try S3/local file first (training data is stored in parquet, not database)
    try:
        df = load_batting_data()
        years = sorted(df["Season"].unique().tolist()) if "Season" in df.columns else []
        
        # Check if we have complete data (should include 2016)
        # If local file is outdated, use hardcoded values
        if years and min(years) > 2016:
            print(f"/stats: Local file has incomplete data (years: {years}), using hardcoded values")
            return EXPECTED_STATS
        
        return {
            "total_player_seasons": len(df),
            "unique_players": df["Name"].nunique() if "Name" in df.columns else 0,
            "years": years
        }
    except Exception as e:
        print(f"/stats: All data sources failed: {e}")
        return EXPECTED_STATS

@app.get("/metrics")
def get_metrics(stat: str, model: str):
    """
        Retrieve model performance metrics (MAE, R2) for a specified stat/model
        Ex: /metrics?stat=HR&model=XGBoost
    """
    table_name = f"{stat.lower()}_{model.lower()}_metrics"

    q = text(f"""
        SELECT * 
        FROM {table_name}
        LIMIT 1
    """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(q)
            row = result.fetchone()
            if row:
                metrics = dict(row._mapping)
                return {
                    "stat": stat,
                    "model": model,
                    "MAE": metrics.get("MAE"),
                    "R2": metrics.get("R2"),
                    "Num_Players": metrics.get("Num_Players")
                }
            else:
                raise HTTPException(status_code=404, detail="Metrics not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/importance")
def get_importance(stat: str, model: str):
    """
        Retrieve feature importance data for a specified stat/model
        Ex: /importance?stat=OPS&model=XGBoost
    """
    import boto3
    from io import BytesIO
    import pandas as pd
    
    # Map model names to match file naming convention
    model_map = {
        "xgboost": "XGBoost",
        "randomforest": "RandomForest", 
        "linearregression": "LinearRegression",
        "ridge": "Ridge"
    }
    
    model_key = model_map.get(model.lower(), model)
    s3_uri = f"s3://mlb-ml-data/models/importance_{stat.upper()}_{model_key}.parquet"
    
    try:
        s3 = boto3.client('s3')
        path = s3_uri[5:]
        bucket, key = path.split("/", 1)
        
        buffer = BytesIO()
        s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
        buffer.seek(0)
        df = pd.read_parquet(buffer)
        
        # Sort by importance and return top features
        df = df.sort_values("Importance", ascending=False)
        
        return {
            "stat": stat,
            "model": model,
            "features": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load importance data: {str(e)}")


@app.get("/players")
def get_players():
    """
        Retrieve list of all unique players with their latest info
        Used for the player search dropdown
    """
    try:
        # Get unique player names from predictions table
        q = text("""
            SELECT DISTINCT "Player"
            FROM ops_linearregression_predictions
            ORDER BY "Player"
        """)
        
        with engine.connect() as conn:
            result = conn.execute(q)
            player_names = [row._mapping["Player"] for row in result]
        
        # Load raw batting data using helper with S3 + local fallbacks
        df = load_batting_data()
        
        # Get latest season data for each player
        latest_df = df.sort_values("Season", ascending=False).drop_duplicates("Name", keep="first")
        
        players = []
        for name in player_names:
            player_data = latest_df[latest_df["Name"] == name]
            if not player_data.empty:
                row = player_data.iloc[0]
                players.append({
                    "Player": name,
                    "Team": row.get("Team", "N/A"),
                    "Age": int(row.get("Age", 0)) if pd.notna(row.get("Age")) else None,
                    "PA": int(row.get("PA", 0)) if pd.notna(row.get("PA")) else None
                })
            else:
                players.append({
                    "Player": name,
                    "Team": "N/A",
                    "Age": None,
                    "PA": None
                })
        
        return {
            "count": len(players),
            "players": players
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/player-history/{player_name}")
def get_player_history(player_name: str):
    """
        Retrieve historical OPS data for a player across seasons
        Returns actual OPS from raw batting data for the chart
    """
    try:
        # Load raw batting data using helper with S3 + local fallbacks
        df = load_batting_data()
        
        # Filter for the player (case-insensitive partial match)
        player_df = df[df["Name"].str.lower().str.contains(player_name.lower())]
        
        if player_df.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Get OPS by season, sorted
        history = player_df[["Season", "OPS"]].sort_values("Season")
        
        # Get 2025 prediction for OPS from LinearRegression
        pred_q = text("""
            SELECT "Predicted"
            FROM ops_linearregression_predictions
            WHERE "Player" ILIKE :player
            LIMIT 1
        """)
        
        predicted_2025 = None
        with engine.connect() as conn:
            result = conn.execute(pred_q, {"player": f"%{player_name}%"})
            row = result.fetchone()
            if row:
                predicted_2025 = row._mapping["Predicted"]
        
        return {
            "player": player_name,
            "history": history.to_dict(orient="records"),
            "predicted_2025_ops": predicted_2025
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))