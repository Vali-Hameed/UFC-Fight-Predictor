from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, brier_score_loss
import pickle
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('ufc-master.csv')
except FileNotFoundError:
    print("Error: 'ufc-master.csv' not found. Please ensure the file is in the correct directory.")
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)
df = df[df['Winner'].isin(['Red', 'Blue'])].copy()

# Create the target variable: 1 if Red wins, 0 if Blue wins
df['Winner_encoded'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

# --- Engineered Features ---
df['RedFinishWins'] = df['RedWinsByKO'].fillna(0) + df['RedWinsBySubmission'].fillna(0)
df['BlueFinishWins'] = df['BlueWinsByKO'].fillna(0) + df['BlueWinsBySubmission'].fillna(0)

df['RedTotalFights'] = df['RedWins'].fillna(df['RedFinishWins']) + df['RedLosses'].fillna(0) + df['RedDraws'].fillna(0)
df['BlueTotalFights'] = df['BlueWins'].fillna(df['BlueFinishWins']) + df['BlueLosses'].fillna(0) + df['BlueDraws'].fillna(0)

# Finish rate: proportion of wins by finish (KO or Sub)
df['RedFinishRate'] = np.where(df['RedTotalFights'] > 0, df['RedFinishWins'] / df['RedTotalFights'], 0)
df['BlueFinishRate'] = np.where(df['BlueTotalFights'] > 0, df['BlueFinishWins'] / df['BlueTotalFights'], 0)

# --- Feature definitions ---
# We use per-corner features rather than just differences. This allows the model
# to learn the context (e.g. knowing a 10-fight veteran vs a 1-fight rookie is different 
# than a 20-fight vs 11-fight veteran, even though the difference is 9 in both cases).
numerical_features = [
    'RedCurrentWinStreak', 'BlueCurrentWinStreak',
    'RedCurrentLoseStreak', 'BlueCurrentLoseStreak', 
    'RedTotalRoundsFought', 'BlueTotalRoundsFought',
    'RedTotalTitleBouts', 'BlueTotalTitleBouts',
    'RedWinsByKO', 'BlueWinsByKO',
    'RedWinsBySubmission', 'BlueWinsBySubmission',
    'RedAvgSigStrLanded', 'BlueAvgSigStrLanded',
    'RedAvgSubAtt', 'BlueAvgSubAtt',
    'RedAvgTDLanded', 'BlueAvgTDLanded',
    'RedAge', 'BlueAge',
    'RedHeightCms', 'BlueHeightCms',
    'RedReachCms', 'BlueReachCms',
    'RedLosses', 'BlueLosses',
    'RedTotalFights', 'BlueTotalFights',
    'RedFinishRate', 'BlueFinishRate'
]

# Note: We excluded categorical Stance for simplicity since it had minimal feature importance
# and complexified symmetrization.

# Split the data chronologically (no shuffle)
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

X_train = df_train[numerical_features]
y_train = df_train['Winner_encoded']
X_test = df_test[numerical_features]
y_test = df_test['Winner_encoded']

print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

# Hyperparameters chosen from tuning script:
# GBM d=2 n=300 lr=0.05 without training augmentation gave best symmetrized accuracy
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', GradientBoostingClassifier(
                                   n_estimators=300,
                                   max_depth=2,
                                   learning_rate=0.05,
                                   subsample=0.8,
                                   min_samples_leaf=20,
                                   random_state=42))])

# ---  Training the Model ---
print("\nTraining the Gradient Boosting model...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Save model and artifacts
tModel = model_pipeline
other_artifacts = {
    "numerical_features": numerical_features,
    "categorical_features": [], # Removed stance
    "data_for_lookups": df
}
with open('ufc_gradient_boosting_model.pkl', 'wb') as f_model:
    pickle.dump(tModel, f_model)

with open('ufc_other_artifacts.pkl', 'wb') as f_artifacts:
    pickle.dump(other_artifacts, f_artifacts)

# ---  Evaluating the Model ---
# ============================================================
# IMPORTANT: Since this model uses per-corner features, it will 
# naturally learn the "Red Corner is the favorite" bias present
# in the training data. This is good for raw predictive power,
# but bad for our website where corners are assigned randomly.
# 
# To solve this, inference MUST be symmetrized (predict forward,
# then swap Red/Blue features, predict reverse, and average).
# We evaluate the test set using this symmetrized approach.
# ============================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# 1. Raw model accuracy (biased)
y_pred = model_pipeline.predict(X_test)
raw_accuracy = accuracy_score(y_test, y_pred)
baseline_accuracy = y_test.mean()  # Red win rate in test set
print(f"\nBaseline accuracy (always predict Red): {baseline_accuracy:.4f}")
print(f"Raw model accuracy (biased): {raw_accuracy:.4f}")

# 2. Symmetrized prediction accuracy (matches production system)
print("\n--- Symmetrized Evaluation (production-style) ---")
forward_proba = model_pipeline.predict_proba(X_test)

# Create reversed features dataframe
X_test_reversed = X_test.copy()
for col in numerical_features:
    if col.startswith('Red'):
        partner = col.replace('Red', 'Blue', 1)
        if partner in numerical_features:
            X_test_reversed[col], X_test_reversed[partner] = X_test[partner].values, X_test[col].values

reverse_proba = model_pipeline.predict_proba(X_test_reversed)

# Average probabilities: P(Red) = average of forward P(Red) and reverse P(Blue)
sym_red_win_prob = (forward_proba[:, 1] + reverse_proba[:, 0]) / 2.0
sym_pred = (sym_red_win_prob > 0.5).astype(int)

sym_accuracy = accuracy_score(y_test, sym_pred)
print(f"Symmetrized accuracy: {sym_accuracy:.4f}")

# Brier score (lower is better)
sym_brier = brier_score_loss(y_test, sym_red_win_prob)
baseline_brier = brier_score_loss(y_test, np.full(len(y_test), baseline_accuracy))
print(f"\nSymmetrized Brier score: {sym_brier:.4f}")
print(f"Baseline Brier score: {baseline_brier:.4f}")

# Classification report
print("\nClassification Report (Symmetrized):")
print(classification_report(y_test, sym_pred, target_names=['Blue Wins', 'Red Wins']))

# Confusion matrix
print("Confusion Matrix (Symmetrized):")
cm = confusion_matrix(y_test, sym_pred)
print(cm)
print(f"\nConfusion Matrix Interpretation:")
print(f"Correctly predicted 'Blue Wins': {cm[0][0]}")
print(f"Incorrectly predicted 'Red Wins' (False Positive): {cm[0][1]}")
print(f"Incorrectly predicted 'Blue Wins' (False Negative): {cm[1][0]}")
print(f"Correctly predicted 'Red Wins': {cm[1][1]}")

# Feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (Top 15)")
print("=" * 60)
clf = model_pipeline.named_steps['classifier']
importances = clf.feature_importances_
feature_importance = sorted(zip(numerical_features, importances), key=lambda x: x[1], reverse=True)
for name, imp in feature_importance[:15]:
    print(f"  {name:35s} importance={imp:.4f}")