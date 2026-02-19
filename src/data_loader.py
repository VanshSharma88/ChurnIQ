"""
data_loader.py
--------------
This file is responsible for:
1. Reading the CSV dataset file
2. Creating the 'Churn' column (our prediction target)
3. Defining which columns are used as features for the ML model
"""

# We import pandas, a library that helps us work with tables (like Excel in Python)
import pandas as pd


def load_data(source):
    """
    This function loads the dataset from a CSV file.

    - 'source' can be a file path (string) OR an uploaded file from Streamlit
    - It returns a pandas DataFrame (a table of data)
    - If something goes wrong, it returns None
    """

    try:
        # pd.read_csv() reads the CSV file and turns it into a table (DataFrame)
        df = pd.read_csv(source)
    except Exception:
        # If the file can't be read, return nothing (None)
        return None

    # Check if the dataset has the 'EngagementLevel' column (we need it to create Churn)
    if 'EngagementLevel' in df.columns:
        # Create a new column called 'Churn'
        # Rule: If EngagementLevel is 'Low' → Churn = 1 (player is leaving)
        #       If EngagementLevel is 'Medium' or 'High' → Churn = 0 (player is staying)
        df['Churn'] = df['EngagementLevel'].apply(lambda x: 1 if x == 'Low' else 0)
    else:
        # If the column doesn't exist, we can't create Churn, so return None
        return None

    # Return the processed table
    return df


def get_feature_lists():
    """
    This function returns two lists:
    1. numerical_features  → columns with numbers (e.g., Age, PlayTimeHours)
    2. categorical_features → columns with text/categories (e.g., Gender, Location)

    These lists tell our ML model WHICH columns to use for learning.
    'PlayerID' and 'EngagementLevel' are excluded because:
    - PlayerID is just an ID (meaningless for prediction)
    - EngagementLevel is what we used to CREATE Churn (so we can't use it again)
    """

    # These columns contain numbers — the model will scale them
    numerical_features = [
        'Age',                       # How old the player is
        'PlayTimeHours',             # Total hours played in the game
        'InGamePurchases',           # Whether the player made purchases (0 = No, 1 = Yes)
        'SessionsPerWeek',           # How many times per week the player plays
        'AvgSessionDurationMinutes', # Average length of each play session (in minutes)
        'PlayerLevel',               # Current level of the player in the game
        'AchievementsUnlocked'       # How many in-game achievements the player has
    ]

    # These columns contain text/categories — the model will one-hot encode them
    categorical_features = [
        'Gender',          # Male or Female
        'Location',        # Country/region (e.g., USA, Europe, Asia)
        'GameGenre',       # Type of game (e.g., Action, RPG, Sports)
        'GameDifficulty'   # Difficulty setting (Easy, Medium, Hard)
    ]

    # Return both lists so other files can use them
    return numerical_features, categorical_features
