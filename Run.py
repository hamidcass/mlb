from dotenv import load_dotenv
import os

load_dotenv()

#import all steps to our pipeline
from ingestion.ingest_stats import run_ingestion
from preprocessing.build_features import run_build_features
from training.train_models import train_all_models
from evalution.evaluate_models import run_eval

#define our s3 paths



RAW_DATA_URI = "s3://mlb-ml-data/raw/batting.parquet"
FEATURES_URI = "s3://mlb-ml-data/prepared/features.parquet"

MODELS_URI = "s3://mlb-ml-data/models"

BASE_MODEL_URI = "s3://mlb-ml-data/models"


MODEL_URIS = {
    "LinearRegression": "s3://mlb-ml-data/models/Linear_Regression.pkl",
    "Ridge": "s3://mlb-ml-data/models/Ridge.pkl",
    "RandomForest": "s3://mlb-ml-data/models/Random_Forest.pkl",
    "XGBoost": "s3://mlb-ml-data/models/XGBoost.pkl"
}

METRICS_URIS = {
    "LinearRegression": "s3://mlb-ml-data/models/metrics_LinearRegression.json",
    "Ridge": "s3://mlb-ml-data/models/metrics_Ridge.json",
    "RandomForest": "s3://mlb-ml-data/models/metrics_RandomForest.json",
    "XGBoost": "s3://mlb-ml-data/models/metrics_XGBoost.json"
}

IMPORTANCE_URIS = {
    "LinearRegression": "s3://mlb-ml-data/models/importance_LinearRegression.parquet",
    "Ridge": "s3://mlb-ml-data/models/importance_Ridge.parquet",
    "RandomForest": "s3://mlb-ml-data/models/importance_RandomForest.parquet",
    "XGBoost": "s3://mlb-ml-data/models/importance_XGBoost.parquet"
}

EVAL_OUTPUT_URI = "s3://mlb-ml-data/evaluation"

TARGET_STATS = ["HR", "AVG", "OPS", "wRC+"]



# #ingest raw data
# run_ingestion(
#     start_year=2020,
#     end_year=2025,
#     min_pa=200,
#     # output_uri="data/raw/batting.parquet",
#     output_uri=RAW_DATA_URI
# )

# for stat in TARGET_STATS:
#     print(f"Building features for {stat}...")

#     run_build_features(
#         target_stat=stat,
#         input_uri=RAW_DATA_URI,
#         output_uri=f"s3://mlb-ml-data/prepared/features_{stat}.parquet"
#     )

# for stat in TARGET_STATS:
#     print(f"\nTraining models for {stat}...")

#     feature_uri = f"s3://mlb-ml-data/prepared/features_{stat}.parquet"

#     model_uris = {
#         "LinearRegression": f"{BASE_MODEL_URI}/{stat}_LinearRegression.pkl",
#         "Ridge": f"{BASE_MODEL_URI}/{stat}_Ridge.pkl",
#         "RandomForest": f"{BASE_MODEL_URI}/{stat}_RandomForest.pkl",
#         "XGBoost": f"{BASE_MODEL_URI}/{stat}_XGBoost.pkl"
#     }

#     metrics_uris = {
#         "LinearRegression": f"{BASE_MODEL_URI}/metrics_{stat}_LinearRegression.json",
#         "Ridge": f"{BASE_MODEL_URI}/metrics_{stat}_Ridge.json",
#         "RandomForest": f"{BASE_MODEL_URI}/metrics_{stat}_RandomForest.json",
#         "XGBoost": f"{BASE_MODEL_URI}/metrics_{stat}_XGBoost.json"
#     }

#     importance_uris = {
#         "LinearRegression": f"{BASE_MODEL_URI}/importance_{stat}_LinearRegression.parquet",
#         "Ridge": f"{BASE_MODEL_URI}/importance_{stat}_Ridge.parquet",
#         "RandomForest": f"{BASE_MODEL_URI}/importance_{stat}_RandomForest.parquet",
#         "XGBoost": f"{BASE_MODEL_URI}/importance_{stat}_XGBoost.parquet"
#     }

#     train_all_models(
#         input_uri=feature_uri,
#         target_stat=stat,
#         model_uris=model_uris,
#         metrics_uris=metrics_uris,
#         importance_uris=importance_uris
#     )

for stat in TARGET_STATS:
    print(f"Evaluating models for {stat}...")

    run_eval(
        models_uri=BASE_MODEL_URI,
        features_uri=f"s3://mlb-ml-data/prepared/features_{stat}.parquet",
        output_uri=f"s3://mlb-ml-data/evaluation/{stat}",
        target_stat=stat
    )

print("Pipeline complete.")
# print("Evaluation Results:", eval_results)