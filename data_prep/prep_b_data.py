import pandas as pd

#we need to prep the data so that the model can notice patterns to train off

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
    elif stat == "wRC":
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
    else:
        return None



def prep_data_hr(dataset, inputs):
    #current year (input metrics) -> following year (output metric)
    
    #step 1: sort by player name and season so we can locate the next year quickly
    dataset = dataset.sort_values(['Name', "Season"])
    #print(dataset)

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
                    row = {
                        "Name": player,
                        "Current_Season": curr_season["Season"],
                        "Next_Season": following_season["Season"],
                    }

                    #add input metrics to row
                    for metric in inputs:
                        row[f"Current_{metric}"] = curr_season[metric]
                        row[f"Target_{metric}"] = following_season[metric]

                    machine_learning_dataset.append(row)
    new_df = pd.DataFrame(machine_learning_dataset)
    return new_df



    pass

# data = pd.read_csv("../data_collection/batting.csv")

# target_stat = "AVG"

# #get input metrics
# input_data = get_input_metrics(target_stat)

# if input_data:
#     ml_dataset = prep_data_hr(data, input_data)
#     print(f"ML Dataset shape: {ml_dataset.shape}")
#     print(f"Number of player-season pairs: {len(ml_dataset)}")

#     # Save
#     ml_dataset.to_csv('prepared_data.csv', index=False)
#     print("\n✓ ML data saved to prepared_data.csv")

#     # Preview
#     print("\nSample ML data:")
#     print(ml_dataset.head())

def run(stat):
    data = pd.read_csv("data_collection/batting.csv")

    target_stat = stat

    #get input metrics
    input_data = get_input_metrics(target_stat)

    if input_data:
        ml_dataset = prep_data_hr(data, input_data)
        print(f"ML Dataset shape: {ml_dataset.shape}")
        print(f"Number of player-season pairs: {len(ml_dataset)}")

        # Save
        ml_dataset.to_csv('prepared_data.csv', index=False)
        print("\n✓ ML data saved to prepared_data.csv")

        # Preview
        print("\nSample ML data:")
        print(ml_dataset.head())

    return ""
        
