import pandas as pd
import numpy as np
import difflib

# ---  Making Predictions on Hypothetical Fights ---

def get_closest_fighter_name(fighter_name, dataframe):
    """Finds the exact or closest fighter name in the dataset."""
    all_fighters = set(dataframe['RedFighter']).union(set(dataframe['BlueFighter']))
    all_fighters = [str(f) for f in all_fighters if pd.notna(f)]
    
    if fighter_name in all_fighters:
        return fighter_name
        
    matches = difflib.get_close_matches(fighter_name, all_fighters, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    return None

def get_latest_stats(fighter_name, dataframe):
    """Finds the most recent fight record for a given fighter."""
    fighter_fights = dataframe[(dataframe['RedFighter'] == fighter_name) | (dataframe['BlueFighter'] == fighter_name)]
    if fighter_fights.empty:
        return None
    return fighter_fights.sort_values(by='Date', ascending=False).iloc[0]

def predict_hypothetical_fight(red_fighter_name, blue_fighter_name, model, dataframe, feature_cols):
    """Predicts the outcome of a hypothetical fight using symmetrized inference over per-corner features."""
    original_red = red_fighter_name
    original_blue = blue_fighter_name
    
    red_matched_name = get_closest_fighter_name(red_fighter_name, dataframe)
    blue_matched_name = get_closest_fighter_name(blue_fighter_name, dataframe)

    if not red_matched_name:
        print(f"Could not find data or close match for {red_fighter_name}")
        return None
    if not blue_matched_name:
        print(f"Could not find data or close match for {blue_fighter_name}")
        return None
        
    if red_matched_name != red_fighter_name:
        print(f"Using closest match '{red_matched_name}' for '{red_fighter_name}'")
        red_fighter_name = red_matched_name
    if blue_matched_name != blue_fighter_name:
        print(f"Using closest match '{blue_matched_name}' for '{blue_fighter_name}'")
        blue_fighter_name = blue_matched_name

    print(f"\n--- Predicting: {red_fighter_name} (Red) vs. {blue_fighter_name} (Blue) ---")

    red_stats_row = get_latest_stats(red_fighter_name, dataframe)
    blue_stats_row = get_latest_stats(blue_fighter_name, dataframe)

    if red_stats_row is None or blue_stats_row is None:
        print("Missing data for one or both fighters.")
        return None

    # Determine which corner each fighter was in during their last fight
    red_corner = 'Red' if red_stats_row['RedFighter'] == red_fighter_name else 'Blue'
    blue_corner = 'Red' if blue_stats_row['RedFighter'] == blue_fighter_name else 'Blue'

    def safe_val(row, col, default=0):
        val = row.get(col)
        if pd.isna(val):
            return default
        return float(val)

    fight_data = {}

    # Map features for the requested Red fighter into the "Red..." feature columns
    # and the requested Blue fighter into the "Blue..." feature columns
    for stat in ['CurrentWinStreak', 'CurrentLoseStreak', 'TotalRoundsFought', 'TotalTitleBouts',
                 'WinsByKO', 'WinsBySubmission', 'AvgSigStrLanded', 'AvgSubAtt', 'AvgTDLanded',
                 'Age', 'HeightCms', 'ReachCms', 'Losses']:
        fight_data[f'Red{stat}'] = safe_val(red_stats_row, f'{red_corner}{stat}')
        fight_data[f'Blue{stat}'] = safe_val(blue_stats_row, f'{blue_corner}{stat}')

    # Engineered features
    for f_name, stats_row, corner, out_prefix in [
        (red_fighter_name, red_stats_row, red_corner, 'Red'),
        (blue_fighter_name, blue_stats_row, blue_corner, 'Blue')
    ]:
        ko = safe_val(stats_row, f'{corner}WinsByKO')
        sub = safe_val(stats_row, f'{corner}WinsBySubmission')
        wins = safe_val(stats_row, f'{corner}Wins')
        losses = safe_val(stats_row, f'{corner}Losses')
        draws = safe_val(stats_row, f'{corner}Draws')
        
        finish_wins = ko + sub
        total_fights = (wins if wins > 0 else finish_wins) + losses + draws
        finish_rate = finish_wins / total_fights if total_fights > 0 else 0
        
        fight_data[f'{out_prefix}TotalFights'] = total_fights
        fight_data[f'{out_prefix}FinishRate'] = finish_rate

    # Ensure columns match model's expected features
    fight_df = pd.DataFrame([fight_data], columns=feature_cols)

    # --- Symmetrized prediction ---
    # The model was trained on data where the Red corner is usually the favorite.
    # To remove this bias for random website matchups, we predict both ways and average.
    
    # 1. Forward prediction: what if Red is Red, and Blue is Blue?
    forward_proba = model.predict_proba(fight_df)[0]  # [P(Blue), P(Red)]

    # 2. Reverse prediction: what if Blue is Red, and Red is Blue?
    reversed_data = fight_data.copy()
    for col in feature_cols:
        if col.startswith('Red'):
            partner = col.replace('Red', 'Blue', 1)
            if partner in feature_cols:
                reversed_data[col], reversed_data[partner] = fight_data[partner], fight_data[col]
    
    reversed_df = pd.DataFrame([reversed_data], columns=feature_cols)
    reverse_proba = model.predict_proba(reversed_df)[0]  # [P(Blue'), P(Red')]

    # Average: forward P(Red wins) with reverse P(Blue wins)
    red_win_prob = (forward_proba[1] + reverse_proba[0]) / 2.0
    blue_win_prob = 1.0 - red_win_prob

    winner = original_red if red_win_prob > blue_win_prob else original_blue
    
    print(f"Prediction Probabilities:")
    print(f"  - {original_blue} (Blue) wins: {blue_win_prob:.2%}")
    print(f"  - {original_red} (Red) wins: {red_win_prob:.2%}")
    print(f"\nPredicted Winner: {winner}")
    
    return {
        "predicted_winner": winner,
        "red_win_prob": float(red_win_prob),
        "blue_win_prob": float(blue_win_prob),
        "red_fighter": original_red,
        "blue_fighter": original_blue
    }
