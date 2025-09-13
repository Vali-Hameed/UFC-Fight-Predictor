import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pickle

try:
    df = pd.read_csv('ufc-master.csv')
except FileNotFoundError:
    print("Error: 'ufc-master.csv' not found. Please ensure the file is in the correct directory.")
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Winner'].isin(['Red', 'Blue'])]

# Create the target variable: 1 if Red wins, 0 if Blue wins
df['Winner_encoded'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

numerical_features = [
    'RedOdds', 'BlueOdds', 'RedAge', 'BlueAge', 'HeightDif', 'ReachDif',
    'WinStreakDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif',
    'KODif', 'SubDif', 'AvgTDDif', 'AvgSubAttDif'
]
categorical_features = ['RedStance', 'BlueStance']
X = df[numerical_features + categorical_features]
y = df['Winner_encoded']
# fill all numerical columns at once
numerical_medians = X[numerical_features].median()
X.loc[:, numerical_features] = X[numerical_features].fillna(numerical_medians)
# fill all categorical columns at once
X.loc[:, categorical_features] = X[categorical_features].fillna('Unknown')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(max_iter=1000, random_state=42))])
# ---  Training the Model ---
print("Training the logistic regression model...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")
tModel=model_pipeline
joblib.dump(tModel, 'ufc_logistic_model.pkl')
# ---  Evaluating the Model ---
print("\n--- Model Evaluation ---")
y_pred = model_pipeline.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Print the classification report for more detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Blue Wins', 'Red Wins']))

# Print the confusion matrix to see true vs. predicted outcomes
print("\nConfusion Matrix:")
# [[True Negative, False Positive], [False Negative, True Positive]]
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Interpretation of Confusion Matrix
print("\nConfusion Matrix Interpretation:")
print(f"Correctly predicted 'Blue Wins': {cm[0][0]}")
print(f"Incorrectly predicted 'Red Wins' (False Positive): {cm[0][1]}")
print(f"Incorrectly predicted 'Blue Wins' (False Negative): {cm[1][0]}")
print(f"Correctly predicted 'Red Wins': {cm[1][1]}")

# ---  Making Predictions on Hypothetical Fights ---
def get_latest_stats(fighter_name, dataframe):
    """Finds the most recent fight record for a given fighter."""
    # Find all fights where the fighter was in either corner
    fighter_fights = dataframe[(dataframe['RedFighter'] == fighter_name) | (dataframe['BlueFighter'] == fighter_name)]

    if fighter_fights.empty:
        return None

    # Return the row of the most recent fight
    return fighter_fights.sort_values(by='Date', ascending=False).iloc[0]

def predict_hypothetical_fight(red_fighter_name, blue_fighter_name, model, dataframe, feature_cols):
    """Predicts the outcome of a hypothetical fight."""
    print(f"\n--- Predicting: {red_fighter_name} (Red) vs. {blue_fighter_name} (Blue) ---")

    red_stats_row = get_latest_stats(red_fighter_name, dataframe)
    blue_stats_row = get_latest_stats(blue_fighter_name, dataframe)

    if red_stats_row is None:
        print(f"Could not find data for {red_fighter_name}")
        return
    if blue_stats_row is None:
        print(f"Could not find data for {blue_fighter_name}")
        return

    # Determine which corner the fighter was in during their last fight to get correct stats
    red_corner = 'Red' if red_stats_row['RedFighter'] == red_fighter_name else 'Blue'
    blue_corner = 'Red' if blue_stats_row['RedFighter'] == blue_fighter_name else 'Blue'

    # Create a dictionary to hold the features for our hypothetical fight
    hypothetical_fight_data = {}

    # --- Extract and calculate features ---
    # Odds (Using average as a neutral baseline since we don't know the real odds)
    hypothetical_fight_data['RedOdds'] = dataframe['RedOdds'].mean()
    hypothetical_fight_data['BlueOdds'] = dataframe['BlueOdds'].mean()

    # Physical attributes
    hypothetical_fight_data['RedAge'] = red_stats_row[f'{red_corner}Age']
    hypothetical_fight_data['BlueAge'] = blue_stats_row[f'{blue_corner}Age']
    red_height = red_stats_row[f'{red_corner}HeightCms']
    blue_height = blue_stats_row[f'{blue_corner}HeightCms']
    red_reach = red_stats_row[f'{red_corner}ReachCms']
    blue_reach = blue_stats_row[f'{blue_corner}ReachCms']

    # Stances
    hypothetical_fight_data['RedStance'] = red_stats_row[f'{red_corner}Stance']
    hypothetical_fight_data['BlueStance'] = blue_stats_row[f'{blue_corner}Stance']

    # Calculate difference features
    hypothetical_fight_data['HeightDif'] = red_height - blue_height
    hypothetical_fight_data['ReachDif'] = red_reach - blue_reach
    hypothetical_fight_data['WinStreakDif'] = red_stats_row[f'{red_corner}CurrentWinStreak'] - blue_stats_row[f'{blue_corner}CurrentWinStreak']
    hypothetical_fight_data['LossDif'] = red_stats_row[f'{red_corner}Losses'] - blue_stats_row[f'{blue_corner}Losses']
    hypothetical_fight_data['TotalRoundDif'] = red_stats_row[f'{red_corner}TotalRoundsFought'] - blue_stats_row[f'{blue_corner}TotalRoundsFought']
    hypothetical_fight_data['TotalTitleBoutDif'] = red_stats_row[f'{red_corner}TotalTitleBouts'] - blue_stats_row[f'{blue_corner}TotalTitleBouts']
    hypothetical_fight_data['KODif'] = red_stats_row[f'{red_corner}WinsByKO'] - blue_stats_row[f'{blue_corner}WinsByKO']
    hypothetical_fight_data['SubDif'] = red_stats_row[f'{red_corner}WinsBySubmission'] - blue_stats_row[f'{blue_corner}WinsBySubmission']
    hypothetical_fight_data['AvgTDDif'] = red_stats_row[f'{red_corner}AvgTDLanded'] - blue_stats_row[f'{blue_corner}AvgTDLanded']
    hypothetical_fight_data['AvgSubAttDif'] = red_stats_row[f'{red_corner}AvgSubAtt'] - blue_stats_row[f'{blue_corner}AvgSubAtt']

    # Create a DataFrame from the dictionary, ensuring column order matches the model's training data
    fight_df = pd.DataFrame([hypothetical_fight_data], columns=feature_cols)

    # Fill any potential NaNs just in case (e.g., stance)
    for col in categorical_features:
        fight_df.fillna({col:'Unknown'}, inplace=True)

    # --- Make the prediction ---
    prediction_proba = model.predict_proba(fight_df)
    blue_win_prob = prediction_proba[0][0]
    red_win_prob = prediction_proba[0][1]

    winner = red_fighter_name if red_win_prob > blue_win_prob else blue_fighter_name
    
    print(f"Prediction Probabilities:")
    print(f"  - {blue_fighter_name} (Blue) wins: {blue_win_prob:.2%}")
    print(f"  - {red_fighter_name} (Red) wins: {red_win_prob:.2%}")
    print(f"\nPredicted Winner: {winner}")


# Example hypothetical fight prediction
predict_hypothetical_fight('Santiago Luna', 'Lee Quang', model_pipeline, df, numerical_features + categorical_features)
predict_hypothetical_fight('Alexander Hernandez', 'Diego Ferreira', model_pipeline, df, numerical_features + categorical_features)
predict_hypothetical_fight('Kelvin Gastelum', 'Dustin Stoltzfus', model_pipeline, df, numerical_features + categorical_features)
predict_hypothetical_fight('Rafa Garcia', 'Jared Gordon', model_pipeline, df, numerical_features + categorical_features)
predict_hypothetical_fight('Rob Font', 'David Martinez', model_pipeline, df, numerical_features + categorical_features)
predict_hypothetical_fight('Diego Lopes', 'Jean Silva', model_pipeline, df, numerical_features + categorical_features)

