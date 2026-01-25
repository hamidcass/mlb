from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
import os

app = FastAPI(title="MLB Prediction API")

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

@app.get("/predictions/{model_name}")
def get_predictions(model_name: str, limit: int = 20):
    
    """
        Retrieve the latest predictions for a specified model
        Ex: /predictions/XGBoost
    """

    table_name = f"{model_name.lower()}_predictions"

    q = text(f"""
        SELECT * FROM {table_name}
        ORDER BY player_name
        LIMIT :limit
             """)
    
    try:
        with engine.connect() as conn:
            result = conn.execute(q, {"limit": limit})
            rows = [dict(row._mapping) for row in result]
        return {"model": model_name, "count": len(rows), "predictions": rows}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.get("/player/{player_name}")
def get_player_prediction(player_name: str):
    """
        Retrieve all predictions for a specified player across all models
        Ex: /player/Mike Trout
    """

    models = ["linear_regression", "ridge", "random_forest", "xgboost"]
    output = {}

    try:
        with engine.connect() as conn:
            for model in models:
                q = text(f"""
                    SELECT * FROM {model}_predictions
                    WHERE "Player" ILIKE :player
                    ORDER BY "Next_Season" DESC
                    LIMIT 1
                            """)
                result = conn.execute(q, {"player": f"%{player_name}%"})
                row = result.fetchone()
                if row:
                    output[model] = dict(row._mapping)
        if not output:
            raise HTTPException(status_code=404, detail="Player not found")
        return {"player": player_name, "predictions": output}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))