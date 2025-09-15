import pandas as pd



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


    # --- Make the prediction ---
    prediction_proba = model.predict_proba(fight_df)
    blue_win_prob = prediction_proba[0][0]
    red_win_prob = prediction_proba[0][1]

    winner = red_fighter_name if red_win_prob > blue_win_prob else blue_fighter_name
    
    print(f"Prediction Probabilities:")
    print(f"  - {blue_fighter_name} (Blue) wins: {blue_win_prob:.2%}")
    print(f"  - {red_fighter_name} (Red) wins: {red_win_prob:.2%}")
    print(f"\nPredicted Winner: {winner}")
    return {
        "predicted_winner": winner,
        "red_win_prob": float(red_win_prob),
        "blue_win_prob": float(blue_win_prob),
        "red_fighter": red_fighter_name,
        "blue_fighter": blue_fighter_name
    }




