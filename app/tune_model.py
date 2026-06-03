"""Hyperparameter tuning script for the UFC Fight Predictor logistic regression model."""
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.read_csv('ufc-master.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')
df = df[df['Winner'].isin(['Red', 'Blue'])]
df['Winner_encoded'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

numerical_features = [
    'RedOdds', 'BlueOdds', 'RedAge', 'BlueAge', 'HeightDif', 'ReachDif',
    'WinStreakDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif',
    'KODif', 'SubDif', 'AvgTDDif', 'AvgSubAttDif'
]
categorical_features = ['RedStance', 'BlueStance']
X = df[numerical_features + categorical_features]
y = df['Winner_encoded']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression(random_state=42))])

param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    'classifier__solver': ['saga'],
    'classifier__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    'classifier__max_iter': [2000],
    'classifier__class_weight': ['balanced', None]
}

print("Running GridSearchCV... (this may take a minute)")
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Blue Wins', 'Red Wins']))

# Also test different train/test splits with the best params
print("\n--- Testing different train/test splits with best params ---")
for split in [0.2, 0.3, 0.4, 0.5, 0.6]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=split, shuffle=False)
    best_model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, best_model.predict(X_te))
    print(f"  test_size={split}: Accuracy={acc:.4f} (train={len(X_tr)}, test={len(X_te)})")
