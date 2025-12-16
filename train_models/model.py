import pandas as pd
import numpy as np

#ML models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

'''
Models used:

LINEAR REGRESSION (Baseline)
    + Fastest, Simple
    - Assumes linear
    -Outliers might make results worse

RANDOM FOREST
    + non-linear relationships
    +outliers wont mess up
    - slower

RIDGE (Linear + Regularization)
    + Linear but better against outliers

'''

def get_stats(stat):
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

def get_features(training_data, stats, target_stat):
    x_cols = [f"Current_{stat}" for stat in stats]
    y_cols = [f"Target_{target_stat}"]
    x = training_data[x_cols]
    y = training_data[y_cols]
    return x, y


def get_target_data(training_data):
    target_data = training_data[training_data["Next_Season"] == 2025].copy()
    training_data = training_data[training_data["Next_Season"] != 2025].copy()
    print(f"Test data: {len(target_data)} rows")
    print(f"Training data: {len(training_data)} rows (2025 removed)")
    return target_data, training_data

def lin_reg(x_train, y_train, x_test, y_test, test_data, target_stat):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("\n✓ MODEL TRAINED")

    predictions = model.predict(x_test).flatten()
    y_test_values = y_test.values.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_values, predictions)

    r2 = r2_score(y_test_values, predictions)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} HR")
    print(f"R² Score:                  {r2:.3f}")
    print(f"\nInterpretation: On average, predictions are off by {mae:.2f} home runs")

    for i in range(min(5, len(predictions))):
        player = test_data.iloc[i]['Name']
        pred = predictions[i]
        actual = y_test_values[i]
        
        # Handle if pred/actual are arrays instead of scalars
        if hasattr(pred, '__len__'):
            pred = pred[0]  # Extract first element if it's an array
        if hasattr(actual, '__len__'):
            actual = actual[0]
            
        error = abs(pred - actual)

        formats = {
            "HR": "5.1f",
            "AVG": "0.3f"
        }

        print(f"  {player:20s} | Predicted: {pred:{formats.get(target_stat)}} {target_stat} | Actual: {actual:{formats.get(target_stat)}} {target_stat} | Error: {error:4.1f}")

training_data = pd.read_csv("../data_prep/prepared_data.csv") 

#extract 2025 season to be used as comparison
target_data, training_data = get_target_data(training_data)

target_stat = "AVG"
#separate into input and output
stats = get_stats(target_stat)

x_train, y_train = get_features(training_data, stats, target_stat)

x_test, y_test = get_features(target_data, stats, target_stat)


learned_data = lin_reg(x_train, y_train, x_test, y_test, target_data, target_stat) 
