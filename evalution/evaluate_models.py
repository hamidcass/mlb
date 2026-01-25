import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from storage.io import load_dataframe, save_dataframe
import joblib  # For loading saved model pipelines
import boto3
from urllib.parse import urlparse
import os
from storage.db import write_df_to_db

results = {}

def download_from_s3(s3_uri, local_path):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)

def evaluate_model(model_pipeline, features_df, target_stat):
    """
    Evaluate a single trained model on the feature df

    model_pipeline: sklearn Pipeline or a single trained model
    features_df: df containing features and target
    target_stat: a string representation of stat: HR, AVG, OPS, WRC+
    """

    #get x and y
    x_cols = [c for c in features_df.columns if c.startswith("Current_")]
    y = features_df[f"Target_{target_stat}"]

    X = features_df[x_cols]

    #predict
    predictions = model_pipeline.predict(X)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)

    #make df of results
    result_df = pd.DataFrame({
        "Player": features_df["Name"],
        "Current_Season": features_df["Current_Season"],
        "Next_Season": features_df["Next_Season"],
        "Actual": y,
        "Predicted": predictions,
        "Error": predictions - y,
        "Abs_Error": np.abs(predictions - y),
        "Pct_Error": (predictions - y) / y * 100
    })

    metrics = {
        "MAE": mae,
        "R2": r2,
        "Num_Players": len(features_df)
    }

    return metrics, result_df

def run_eval(models_uri, features_uri, output_uri, target_stat):
    """
    function used to eval models

    models_uri: local path or s3 where trained pipelines are stored
    features_uri: local path or s3 where prepped features are stored
    output_uri: local path or s3 where eval results are stored
    target_stat: a string representation of stat: HR, AVG, OPS, WRC+
    """

    print(f"Loading features from {features_uri}")
    features_df = load_dataframe(features_uri)

    print(f"Loading trained models from {models_uri}")

    model_files = ["Linear_Regression.pkl", "Ridge.pkl", "Random_Forest.pkl", "XGBoost.pkl"]
 
    for model_file in model_files:
        model_name = model_file.replace(".pkl", "")
        print(f"Evaluating model: {model_name}")

        # model_pipeline = joblib.load(f"{models_uri}/{model_file}")
        s3_model_uri = f"{models_uri}/{model_file}"
        local_model_path = f"/tmp/{model_file}"

        download_from_s3(s3_model_uri, local_model_path)
        model_pipeline = joblib.load(local_model_path)

        metrics, result_df = evaluate_model(model_pipeline, features_df, target_stat)

        #save results for the given model
        # save_dataframe(result_df, f"{output_uri}/{model_name}_predictions.parquet")

        #write to db
        write_df_to_db(result_df, f"{model_name.lower()}_predictions")

        #store metrics in dict
        results[model_name] = metrics
        metrics_df = pd.DataFrame([metrics])
        write_df_to_db(metrics_df, f"{model_name.lower()}_metrics")

        print(f"{model_name} - MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")

    return results