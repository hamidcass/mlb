import pandas as pd
import boto3
from io import BytesIO


#create s3 client which will read env file by itself
s3 = boto3.client('s3')

def save_dataframe(df, uri):
    """
    Save dataframe locally or to s3
    
    uri examples:
        local: data/raw/batting.parquet
        s3: "s3://mlb-ml-data/raw/batting.parquet"
    """
    if uri.startswith("s3://"):
        
        #parse bucket and key
        path = uri[5:]
        bucket, key = path.split("/", 1)

        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        s3.upload_fileobj(buffer, Bucket=bucket, Key=key)
        print(f"Data saved to S3 at {uri}")

    else:
        df.to_parquet(uri)
        print(f"Data saved locally at {uri}")

def load_dataframe(uri):
    """
    Load df locally or from s3
    
    """
    if uri.startswith("s3://"):
        
        path = uri[5:]
        bucket, key = path.split("/", 1)

        buffer = BytesIO()
        s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=buffer)
        buffer.seek(0)
        df = pd.read_parquet(buffer)
        print(f"Data loaded from S3 at {uri}")
        return df

    else:
        df = pd.read_parquet(uri)
        print(f"Data loaded locally from {uri}")
        return df