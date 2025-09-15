from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import pandas as pd
try:
    df = pd.read_csv('app\ufc-master.csv')
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
other_artifacts = {
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "data_for_lookups": df
    }
with open('ufc_logistic_model.pkl','wb') as f_model:
    pickle.dump(tModel, f_model)

with open('ufc_other_artifacts.pkl','wb') as f_artifacts:
    pickle.dump(other_artifacts, f_artifacts)

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