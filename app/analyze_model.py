"""Quick analysis script to identify methodology issues in the UFC model."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('ufc-master.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df = df[df['Winner'].isin(['Red', 'Blue'])]
df['Winner_encoded'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

print("=" * 60)
print("UFC MODEL ANALYSIS")
print("=" * 60)

# 1. Class imbalance
total = len(df)
red_wins = (df['Winner'] == 'Red').sum()
blue_wins = (df['Winner'] == 'Blue').sum()
print(f"\n1. CLASS IMBALANCE")
print(f"   Total fights: {total}")
print(f"   Red wins: {red_wins} ({red_wins/total*100:.1f}%)")
print(f"   Blue wins: {blue_wins} ({blue_wins/total*100:.1f}%)")

# 2. Red corner = favorite bias
print(f"\n2. RED CORNER = FAVORITE BIAS")
print(f"   Red is favorite (lower/negative odds): {(df['RedOdds'] < df['BlueOdds']).sum()} ({(df['RedOdds'] < df['BlueOdds']).mean()*100:.1f}%)")
print(f"   Blue is favorite: {(df['RedOdds'] > df['BlueOdds']).sum()} ({(df['RedOdds'] > df['BlueOdds']).mean()*100:.1f}%)")
print(f"   Mean RedOdds: {df['RedOdds'].mean():.1f}")
print(f"   Mean BlueOdds: {df['BlueOdds'].mean():.1f}")

# 3. Train/test split analysis
X = df[['RedOdds']]
y = df['Winner_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
train_dates = df.iloc[:len(X_train)]['Date']
test_dates = df.iloc[len(X_train):]['Date']
print(f"\n3. TRAIN/TEST SPLIT")
print(f"   Train: {len(X_train)} fights ({train_dates.min().date()} to {train_dates.max().date()})")
print(f"   Test: {len(X_test)} fights ({test_dates.min().date()} to {test_dates.max().date()})")
print(f"   Train Red win rate: {y_train.mean()*100:.1f}%")
print(f"   Test Red win rate: {y_test.mean()*100:.1f}%")

# 4. DATA LEAKAGE CHECK - RedOdds and BlueOdds
print(f"\n4. DATA LEAKAGE: BETTING ODDS")
print(f"   RedOdds and BlueOdds are included as features.")
print(f"   These are set by bookmakers who already factor in fighter stats,")
print(f"   records, rankings, and expert analysis.")
print(f"   This means the model is partially 'cheating' by using the")
print(f"   collective wisdom of the betting market as input.")

# Test: how good is a model that ONLY uses odds?
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numerical_features = [
    'RedOdds', 'BlueOdds', 'RedAge', 'BlueAge', 'HeightDif', 'ReachDif',
    'WinStreakDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif',
    'KODif', 'SubDif', 'AvgTDDif', 'AvgSubAttDif'
]

full_X = df[numerical_features]
X_train_full, X_test_full, y_train_f, y_test_f = train_test_split(full_X, y, test_size=0.2, shuffle=False)

# Odds-only model
odds_only = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=42))
])
odds_only.fit(X_train_full[['RedOdds', 'BlueOdds']], y_train_f)
odds_acc = accuracy_score(y_test_f, odds_only.predict(X_test_full[['RedOdds', 'BlueOdds']]))
print(f"\n   Accuracy with ONLY RedOdds + BlueOdds: {odds_acc*100:.1f}%")

# Full model (without odds)
no_odds_features = [f for f in numerical_features if f not in ['RedOdds', 'BlueOdds']]
no_odds = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=42))
])
no_odds.fit(X_train_full[no_odds_features], y_train_f)
no_odds_acc = accuracy_score(y_test_f, no_odds.predict(X_test_full[no_odds_features]))
print(f"   Accuracy WITHOUT odds (fighter stats only): {no_odds_acc*100:.1f}%")

# Full model
full = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, random_state=42))
])
full.fit(X_train_full, y_train_f)
full_acc = accuracy_score(y_test_f, full.predict(X_test_full))
print(f"   Accuracy WITH all features (odds + stats): {full_acc*100:.1f}%")

# 5. PREDICTION SYMMETRY TEST
print(f"\n5. PREDICTION SYMMETRY TEST")
print(f"   In your simulator, Fighter 1 is ALWAYS placed in the Red corner.")
print(f"   But in real life, corner assignment matters because the model")
print(f"   learned that Red wins 62% of the time (Red is usually the favorite).")
print(f"   Swapping fighter order should ideally just flip probabilities,")
print(f"   but because odds are hardcoded to -110/-110, the result may")
print(f"   differ based on which corner a fighter is placed in.")

import pickle
with open('ufc_logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('ufc_other_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

from ufc_predictor import predict_hypothetical_fight
feature_cols = artifacts['categorical_features'] + artifacts['numerical_features']
lookup_df = artifacts['data_for_lookups']

print(f"\n   Test: Conor McGregor vs Khabib Nurmagomedov")
r1 = predict_hypothetical_fight('Conor McGregor', 'Khabib Nurmagomedov', model, lookup_df, feature_cols)
r2 = predict_hypothetical_fight('Khabib Nurmagomedov', 'Conor McGregor', model, lookup_df, feature_cols)
if r1 and r2:
    print(f"\n   McGregor (Red) vs Khabib (Blue): McGregor win prob = {r1['red_win_prob']*100:.1f}%")
    print(f"   Khabib (Red) vs McGregor (Blue): Khabib win prob = {r2['red_win_prob']*100:.1f}%")
    print(f"   Sum of McGregor win probs: {r1['red_win_prob']*100:.1f}% + {r2['blue_win_prob']*100:.1f}% = {(r1['red_win_prob']+r2['blue_win_prob'])*100:.1f}%")
    print(f"   (Should be ~100% if symmetric, but won't be due to corner bias)")

# 6. Feature importance
print(f"\n6. FEATURE IMPORTANCE (Logistic Regression Coefficients)")
clf = model.named_steps['classifier']
preprocessor = model.named_steps['preprocessor']

# Get feature names after preprocessing
num_features = numerical_features
cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(['RedStance', 'BlueStance']).tolist()
all_feature_names = num_features + cat_features

coefficients = clf.coef_[0]
feature_importance = sorted(zip(all_feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
print(f"   Top features driving predictions:")
for name, coef in feature_importance[:10]:
    direction = "Red wins" if coef > 0 else "Blue wins"
    print(f"   {name:30s} coef={coef:+.4f}  (favors {direction})")

# 7. Missing data
print(f"\n7. MISSING DATA IN KEY FEATURES")
for col in numerical_features:
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"   {col}: {missing} missing ({missing/total*100:.1f}%)")

print(f"\n8. HARDCODED ODDS IN SIMULATOR")
print(f"   The simulator hardcodes RedOdds=-110 and BlueOdds=-110.")
print(f"   But if odds are the most important feature, this means the")
print(f"   simulator predictions may be very different from event predictions")
print(f"   that use real betting odds.")
