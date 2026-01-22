from dotenv import load_dotenv


load_dotenv()

from preprocessing.build_features import run_build_features

from ingestion.ingest_stats import run_ingestion




# import os
# print("AWS key loaded:", os.getenv("AWS_ACCESS_KEY_ID") is not None)

# run_ingestion(
#     start_year=2020,
#     end_year=2024,
#     min_pa=200,
#     output_uri="data/raw/batting.parquet"
# )

#get raw data, send to s3
# run_ingestion(
#     start_year=2020,
#     end_year=2024,
#     min_pa=200,
#     output_uri="s3://mlb-ml-data/raw/batting.parquet"
# )

#build features and save to s3
run_build_features(
    target_stat="HR",
    # input_uri="data/raw/batting.parquet",
    # output_uri="data/processed/features.parquet"
    input_uri="s3://mlb-ml-data/raw/batting.parquet",
    output_uri="s3://mlb-ml-data/prepared/features.parquet"
)