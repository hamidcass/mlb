import pandas as pd
from pybaseball import batting_stats
from storage.io import save_dataframe 

def fetch_batting_data(start_year, end_year, min_pa):
    data = []

    for year in range(start_year, end_year + 1):
        print(f"Fetching {year} data...")
        try:
            year_data = batting_stats(year, qual=min_pa)
            year_data['Season'] = year  # add year
            data.append(year_data)
            print(f"  ✓ Found {len(year_data)} players")
        except Exception as e:
            print(f"  ✗ Error fetching {year}: {e}")

    all_years_data = pd.concat(data, ignore_index=True)
    all_years_data['Team'] = all_years_data['Team'].replace("- - -", "MULTI")
    return all_years_data

def filter_multi_year_players(df, min_seasons=2):
    player_count = df['Name'].value_counts()
    multi_year_players = player_count[player_count >= min_seasons]
    filtered_df = df[df['Name'].isin(multi_year_players.index)]
    return filtered_df

def run_ingestion(start_year: int, end_year: int, min_pa: int, output_uri: str):

    print("Starting batting data ingestion...")

    raw_df = fetch_batting_data(start_year, end_year, min_pa)
    filtered_df = filter_multi_year_players(raw_df)

    print(f"\nTotal records after filtering: {len(filtered_df)}")
    print(f"Total unique players after filtering: {filtered_df['Name'].nunique()}")

    save_dataframe(filtered_df, output_uri)

    print(f"Data saved to {output_uri}")
    return output_uri