import pandas as pd
import numpy as np

#ML models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
#returns the appropriate input features based on what the target stat is 
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

#looks in df and extracts the feature column data and the target stat data
#returns: x (2d list: first season features), y (1d list: nect season output stat)
def get_features(training_data, stats, target_stat):
    x_cols = [f"Current_{stat}" for stat in stats]
    y_col = f"Target_{target_stat}"
    x = training_data[x_cols]
    y = training_data[y_col]
    return x, y

#take any rows from 2024->2025 because we dont want 2025 stats in our training data (it will be our test data instead)
def get_target_data(training_data):
    target_data = training_data[training_data["Next_Season"] == 2025].copy()
    training_data = training_data[training_data["Next_Season"] != 2025].copy()
    print(f"Test data: {len(target_data)} rows")
    print(f"Training data: {len(training_data)} rows (2025 removed)")
    return target_data, training_data

#Predicts using Linear regression
def lin_reg(x_train, y_train, x_test, y_test, test_data, target_stat):
    
    
    model = LinearRegression()
    model.fit(x_train, y_train) #give model the features and output
    print("\n***Model has been trained using linear regression***")

    predictions = model.predict(x_test).flatten()

    y_test_values = y_test.values.flatten()


    mae = mean_absolute_error(y_test_values, predictions)

    
    print(f"\n{'='*60}")
    print("LINEAR REGRESSION")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE): {mae:.3f} {target_stat}")
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

        print(f"  {player:20s} | Predicted: {pred:{formats.get(target_stat)}} {target_stat} | Actual: {actual:{formats.get(target_stat)}} {target_stat} | Error: {error:{formats.get(target_stat)}}")

def random_forest(x_train, y_train, x_test, y_test, test_data, target_stat):

    model = RandomForestRegressor(
        n_estimators=100, #how many trees do we want (more = better but slower)
        max_depth=10, #how deep can each tree go (deeper = smarter but can overfit and longer)
        random_state=42, #reproducibility (42 is standard)
    )
    model.fit(x_train, y_train)

    print("\n***Model has been trained using random forest***")

    predictions = model.predict(x_test).flatten()
    y_test_values = y_test.values.flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_values, predictions)

    
    print(f"\n{'='*60}")
    print("RANDOM FOREST")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE): {mae:.3f} {target_stat}")
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

        print(f"  {player:20s} | Predicted: {pred:{formats.get(target_stat)}} {target_stat} | Actual: {actual:{formats.get(target_stat)}} {target_stat} | Error: {error:{formats.get(target_stat)}}")

def ridge(x_train, y_train, x_test, y_test, test_data, target_stat):
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    print("\n***Model has been trained using ridge***")

    predictions = model.predict(x_test).flatten()

    y_test_values = y_test.values.flatten()


    mae = mean_absolute_error(y_test_values, predictions)

    
    print(f"\n{'='*60}")
    print("RIDGE")
    print(f"{'='*60}")
    print(f"Mean Absolute Error (MAE): {mae:.3f} {target_stat}")
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

        print(f"  {player:20s} | Predicted: {pred:{formats.get(target_stat)}} {target_stat} | Actual: {actual:{formats.get(target_stat)}} {target_stat} | Error: {error:{formats.get(target_stat)}}")








training_data = pd.read_csv("../data_prep/prepared_data.csv") 

#extract 2025 season to be used as comparison
target_data, training_data = get_target_data(training_data)

target_stat = "AVG"
#separate into input and output
stats = get_stats(target_stat)

x_train, y_train = get_features(training_data, stats, target_stat)
x_test, y_test = get_features(target_data, stats, target_stat)


learned_data = lin_reg(x_train, y_train, x_test, y_test, target_data, target_stat) 
learned_data = random_forest(x_train, y_train, x_test, y_test, target_data, target_stat) 
learned_data = ridge(x_train, y_train, x_test, y_test, target_data, target_stat) 
