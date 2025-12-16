import pandas as pd




from pybaseball import batting_stats

def fetch_data(min_year, max_year, min_pa=200):
    data = []

    for year in range(min_year, max_year+1):
        print(f"Fetching {year} data...")
        try:
            year_data = batting_stats(year, qual=min_pa)
            year_data['Season'] = year #add year
            data.append(year_data)
            print(f"  ✓ Found {len(year_data)} players")
        except Exception as e:
            print(f"  ✗ Error fetching {year}: {e}")

        ##make one big dataframe from all rows
    all_years_data = pd.concat(data, ignore_index=True)
    return all_years_data


print("Beginning the batting data collection...")
batting_data = fetch_data(2020, 2025, 200)

print(f"\nTotal records: {len(batting_data)}")
print(f"Total unique players: {batting_data['Name'].nunique()}")

#we dont want rookies or 1 szn players
player_count = batting_data['Name'].value_counts()
multi_year_players = player_count[player_count > 1]
print(f"Players with 2+ seasons: {len(multi_year_players)}")

only_multi_year_players = batting_data[batting_data['Name'].isin(multi_year_players.index)]
print(f"Total records after filtering: {len(only_multi_year_players)}")
print(f"Total unique players after filtering: {only_multi_year_players['Name'].nunique()}")

only_multi_year_players.to_csv("batting.csv", index=False)