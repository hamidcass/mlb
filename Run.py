from ingestion.ingest_stats import run_ingestion

run_ingestion(
    start_year=2020,
    end_year=2024,
    min_pa=200,
    output_uri="data/raw/batting.parquet"
)

# S3 run (once bucket exists)
# run_ingestion(
#     start_year=2020,
#     end_year=2024,
#     min_pa=200,
#     output_uri="s3://mlb-ml-data/raw/batting.parquet"
# )