from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import pickle
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('ufc-master.csv')
except FileNotFoundError:
    print("Error: 'ufc-master.csv' not found. Please ensure the file is in the correct directory.")
    exit()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df = df[df['Winner'].isin(['Red', 'Blue'])]

# Create the target variable: 1 if Red wins, 0 if Blue wins
df['Winner_encoded'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

# --- Create symmetric StanceMatchup feature ---
# Sort both stances alphabetically so the matchup is corner-invariant
# e.g. Red=Southpaw, Blue=Orthodox → "Orthodox_vs_Southpaw" (same as if swapped)
def make_stance_matchup(row):
    red_stance = str(row.get('RedStance', 'Unknown')) if pd.notna(row.get('RedStance')) else 'Unknown'
    blue_stance = str(row.get('BlueStance', 'Unknown')) if pd.notna(row.get('BlueStance')) else 'Unknown'
    sorted_stances = sorted([red_stance, blue_stance])
    return f"{sorted_stances[0]}_vs_{sorted_stances[1]}"

df['StanceMatchup'] = df.apply(make_stance_matchup, axis=1)

# --- Feature definitions ---
# All numerical features are DIFFERENCE features (Red - Blue), ensuring symmetry.
# No per-corner features (RedOdds, BlueOdds, RedAge, BlueAge) — these break symmetry
# and odds cause data leakage (bookmakers already encode fighter stats).
numerical_features = [
    'HeightDif', 'ReachDif', 'AgeDif',
    'WinStreakDif', 'LoseStreakDif', 'LongestWinStreakDif',
    'WinDif', 'LossDif',
    'TotalRoundDif', 'TotalTitleBoutDif',
    'KODif', 'SubDif',
    'SigStrDif', 'AvgSubAttDif', 'AvgTDDif'
]
categorical_features = ['StanceMatchup']

X = df[numerical_features + categorical_features]
y = df['Winner_encoded']

# Time-series aware split (no shuffle — prevent future data leaking into training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown_vs_Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(
                                   C=0.01,
                                   penalty='l2',
                                   solver='saga',
                                   max_iter=2000,
                                   random_state=42))])

# ---  Training the Model ---
print("Training the logistic regression model...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# Save model and artifacts
tModel = model_pipeline
other_artifacts = {
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "data_for_lookups": df
}
with open('ufc_logistic_model.pkl', 'wb') as f_model:
    pickle.dump(tModel, f_model)

with open('ufc_other_artifacts.pkl', 'wb') as f_artifacts:
    pickle.dump(other_artifacts, f_artifacts)

# ---  Evaluating the Model ---
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)

# Baseline: always predict Red (majority class)
baseline_accuracy = y_test.mean()  # Red win rate in test set
print(f"\nBaseline accuracy (always predict Red): {baseline_accuracy:.4f}")

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print(f"Improvement over baseline: {(accuracy - baseline_accuracy)*100:+.1f} percentage points")

# Log-loss (better metric for probabilistic predictions)
logloss = log_loss(y_test, y_pred_proba)
baseline_logloss = log_loss(y_test, np.full_like(y_pred_proba, [1 - baseline_accuracy, baseline_accuracy]))
print(f"\nModel log-loss: {logloss:.4f}")
print(f"Baseline log-loss: {baseline_logloss:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Blue Wins', 'Red Wins']))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\nConfusion Matrix Interpretation:")
print(f"Correctly predicted 'Blue Wins': {cm[0][0]}")
print(f"Incorrectly predicted 'Red Wins' (False Positive): {cm[0][1]}")
print(f"Incorrectly predicted 'Blue Wins' (False Negative): {cm[1][0]}")
print(f"Correctly predicted 'Red Wins': {cm[1][1]}")

# Model intercept (should be near zero for a symmetric model)
intercept = model_pipeline.named_steps['classifier'].intercept_[0]
print(f"\nModel intercept: {intercept:.4f}")
print(f"(Should be near 0 for symmetric predictions -- {'OK' if abs(intercept) < 0.3 else 'High bias'})")

# Feature importance
print("\nFeature Importance (Top 10):")
clf = model_pipeline.named_steps['classifier']
preprocessor_fitted = model_pipeline.named_steps['preprocessor']
num_names = numerical_features
cat_names = preprocessor_fitted.transformers_[1][1].named_steps['onehot'].get_feature_names_out(['StanceMatchup']).tolist()
all_feature_names = num_names + cat_names
coefficients = clf.coef_[0]
feature_importance = sorted(zip(all_feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
for name, coef in feature_importance[:10]:
    direction = "Red wins" if coef > 0 else "Blue wins"
    print(f"  {name:35s} coef={coef:+.4f}  (favors {direction})")