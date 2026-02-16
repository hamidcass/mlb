"""
Migration script: Load importance parquet files from local storage into PostgreSQL.
Run this ON EC2 where the database is accessible.

Usage:
    cd ~/InningAI/backend
    source venv/bin/activate
    python migrate_importance_to_db.py
"""

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from storage.db import write_df_to_db

TARGET_STATS = ["HR", "AVG", "OPS", "wRC_PLUS"]
MODELS = ["LinearRegression", "Ridge", "RandomForest", "XGBoost"]

def migrate():
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "data" / "importance"
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        print("Make sure you've run 'git pull' to get the importance parquet files.")
        return
    
    success = 0
    failed = 0
    
    for stat in TARGET_STATS:
        for model in MODELS:
            filename = f"importance_{stat}_{model}.parquet"
            filepath = data_dir / filename
            table_name = f"{stat.lower()}_{model.lower()}_importance"
            
            try:
                df = pd.read_parquet(filepath)
                write_df_to_db(df, table_name)
                print(f"✅ {filename} -> {table_name} ({len(df)} rows)")
                success += 1
            except Exception as e:
                print(f"❌ {filename} -> {e}")
                failed += 1

    print(f"\nMigration complete! {success} succeeded, {failed} failed.")

if __name__ == "__main__":
    migrate()
