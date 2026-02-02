from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
import os

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="MLB Prediction API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
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