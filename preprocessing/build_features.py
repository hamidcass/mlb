import pandas as pd
from storage.io import load_dataframe, save_dataframe


#we need to prep the data so that the model can notice patterns to train off

#team to long name map
team_name_id = {
    #AL EAST
    "NYY": 147,
    "TOR": 141,
    "BAL": 110,
    "TBR": 139,
    "BOS": 111,

    #AL CENTRAL
    "DET": 116,
    "CHW": 145,
    "KCR": 118,
    "MIN": 142, 
    "CLE": 114,


    #AL WEST
    "HOU": 117,
    "SEA": 136,
    "LAA": 108,
    "OAK": 133,
    "TEX": 140,

    #NL EAST
    "NYM": 121,
    "MIA": 146,
    "WSN": 120,
    "PHI": 143,
    "ATL": 144,

    #NL CENTRAL
    "CIN": 113,
    "PIT": 134,
    "CHC": 112,
    "MIL": 158,
    "STL": 138,

    #NL WEST
    "LAD": 119,
    "SDP": 135,
    "COL": 115,
    "ARI": 109,
    "SFG": 137,

}



def get_input_metrics(stat):
    if stat == "HR":
        return [
            'Age',           # Players peak ~27-30, then decline
            'PA',            # Playing time (more PAs = more HR opportunities)
            'HR',            # Last year's HR (best predictor!)
            'ISO',           # Isolated power (SLG - AVG)
            'FB%',           # Flyball rate (more flyballs = more HR potential)
            'HR/FB',         # HR per flyball rate
            'Barrel%',       # Statcast: % of barrels (ideal contact)
            'HardHit%',      # Statcast: hard-hit ball rate
            'EV',            # Exit velocity (harder hit = more HR)
            'Pull%',         # Pull hitters hit more HR
        ]
    elif stat == "AVG":
        return [
            'Age',           # Context
            'PA',            # Playing time
            'AVG',           # Last year's AVG (best predictor!)
            'K%',            # Strikeout rate (fewer K = higher AVG)
            'Contact%',      # Contact rate on swings
            'BABIP',         # Batting average on balls in play
            'LD%',           # Line drive rate (line drives = hits)
            'Hard%',         # Hard-hit ball % (harder = more hits)
            'Soft%',         # Soft contact % (less = better)
            'xBA',           # Expected batting average (Statcast)
        ]
    elif stat == "OPS":
        return [
                'Age',           # Context
                'PA',            # Playing time
                'OPS',           # Last year's OPS (best predictor!)
                'wRC+',          # Weighted runs created (overall value)
                'BB%',           # Walk rate (boosts OBP)
                'K%',            # Strikeout rate
                'ISO',           # Power component
                'BABIP',         # Luck/contact quality
                'HardHit%',      # Quality of contact
                'Barrel%',       # Elite contact
                'xwOBA',         # Expected wOBA (similar to OPS)
        ]
    elif stat == "wRC+":
        return [
            'Age',           # Context
            'PA',            # Playing time
            'wRC+',          # Last year's wRC+ (best predictor!)
            'wOBA',          # Foundation of wRC+
            'BB%',           # Walks
            'K%',            # Strikeouts
            'ISO',           # Power
            'AVG',           # Contact
            'BABIP',         # Luck factor
            'Barrel%',       # Quality contact
            'HardHit%',      # Quality contact
        ]
    elif stat == "WAR":
        return [
            "Age",
            "PA",
            "G",

            "wOBA",
            "wRC+",
            "ISO",
            "BB%",
            "K%",
            "Barrel%",
            "HardHit%",
            "SB",
            

        ]
    else:
        return None


def prep_data(dataset, inputs):

    
    #current year (input metrics) -> following year (output metric)
    
    #step 1: sort by player name and season so we can locate the next year quickly
    dataset = dataset.sort_values(['Name', "Season"])


    machine_learning_dataset = []

    for player, player_data in dataset.groupby("Name"):
        player_data = player_data = player_data.sort_values("Season")


        #Conservative approach: we must only get players who have consecutive years played to account for facotrs like injury recover
        
        #get pairs consecutive seasons for each player (2021->2022, 2022->2025, etc)
        if (len(player_data) >= 2):
            for i in range(len(player_data)-1):
                curr_season = player_data.iloc[i]
                following_season = player_data.iloc[i+1]

                #only include if seasons were back-to-back
                if following_season["Season"] == curr_season["Season"] + 1: 


                    # Handle MULTI-team seasons
                    if curr_season["Team"] == "MULTI":
                       
                       pass


                    row = {
                        "Name": player,
                        "Current_Season": curr_season["Season"],
                        "Next_Season": following_season["Season"],
                        "Current_Team": curr_season["Team"], 
                        "Next_Team": following_season["Team"],
                    }

                    #add input metrics to row
                    for metric in inputs:
                        row[f"Current_{metric}"] = curr_season[metric]
                        row[f"Target_{metric}"] = following_season[metric]

                    machine_learning_dataset.append(row)
    new_df = pd.DataFrame(machine_learning_dataset)
    return new_df






def run_build_features(target_stat: str, input_uri: str, output_uri: str):
    
    """
    Load raw batting data and build ML features and save to parquet
    Works locally or directly to S3
    """
    
    
    
    print("Loading raw data from {input_uri}...")
    data = load_dataframe(input_uri)

    print("Prepping features for target stat: {target_stat}...")
    inputs = get_input_metrics(target_stat)
    if not inputs:
        raise ValueError(f"Unsupported target stat: {target_stat}")
    
    features_df = prep_data(data, inputs)
    print(f"Feature data shape: {features_df.shape}")
    print(f"Number of player-season pairs: {len(features_df)}")

    save_dataframe(features_df, output_uri)
    print(f"Features saved to {output_uri}")
    return features_df


