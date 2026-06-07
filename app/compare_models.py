"""Quick comparison: augmented vs non-augmented model, evaluated symmetrized."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

df = pd.read_csv('ufc-master.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)
df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
df['Winner_encoded'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

df['RedFinishWins'] = df['RedWinsByKO'].fillna(0) + df['RedWinsBySubmission'].fillna(0)
df['BlueFinishWins'] = df['BlueWinsByKO'].fillna(0) + df['BlueWinsBySubmission'].fillna(0)
df['RedTotalFights'] = df['RedWins'].fillna(df['RedFinishWins']) + df['RedLosses'].fillna(0) + df['RedDraws'].fillna(0)
df['BlueTotalFights'] = df['BlueWins'].fillna(df['BlueFinishWins']) + df['BlueLosses'].fillna(0) + df['BlueDraws'].fillna(0)
df['ExperienceDif'] = df['RedTotalFights'] - df['BlueTotalFights']
df['RedFinishRate'] = np.where(df['RedTotalFights'] > 0, df['RedFinishWins'] / df['RedTotalFights'], 0)
df['BlueFinishRate'] = np.where(df['BlueTotalFights'] > 0, df['BlueFinishWins'] / df['BlueTotalFights'], 0)
df['FinishRateDif'] = df['RedFinishRate'] - df['BlueFinishRate']

def make_stance_matchup(row):
    red_stance = str(row.get('RedStance', 'Unknown')) if pd.notna(row.get('RedStance')) else 'Unknown'
    blue_stance = str(row.get('BlueStance', 'Unknown')) if pd.notna(row.get('BlueStance')) else 'Unknown'
    sorted_stances = sorted([red_stance, blue_stance])
    return f"{sorted_stances[0]}_vs_{sorted_stances[1]}"

df['StanceMatchup'] = df.apply(make_stance_matchup, axis=1)

numerical_features = [
    'HeightDif', 'ReachDif', 'AgeDif',
    'WinStreakDif', 'LoseStreakDif', 'LongestWinStreakDif',
    'WinDif', 'LossDif',
    'TotalRoundDif', 'TotalTitleBoutDif',
    'KODif', 'SubDif',
    'SigStrDif', 'AvgSubAttDif', 'AvgTDDif',
    'ExperienceDif', 'FinishRateDif',
]
categorical_features = ['StanceMatchup']

train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

X_test = df_test[numerical_features + categorical_features]
y_test = df_test['Winner_encoded']

def make_pipeline():
    num_t = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_t = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown_vs_Unknown')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    prep = ColumnTransformer([('num', num_t, numerical_features), ('cat', cat_t, categorical_features)])
    return Pipeline([('preprocessor', prep),
                     ('classifier', GradientBoostingClassifier(
                         n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, min_samples_leaf=20, random_state=42))])

def symmetrized_eval(model, X_test, y_test):
    fwd = model.predict_proba(X_test)
    X_rev = X_test.copy()
    for feat in numerical_features:
        X_rev[feat] = -X_rev[feat]
    rev = model.predict_proba(X_rev)
    sym_prob = (fwd[:, 1] + rev[:, 0]) / 2.0
    sym_pred = (sym_prob > 0.5).astype(int)
    return accuracy_score(y_test, sym_pred), brier_score_loss(y_test, sym_prob)

baseline = y_test.mean()
print(f"Baseline (always Red): {baseline:.4f}")
print(f"Test set size: {len(y_test)}, Red wins: {y_test.sum()}, Blue wins: {(1-y_test).sum()}")
print()

# --- Model A: No augmentation ---
print("=== Model A: NO augmentation (original data only) ===")
X_train_a = df_train[numerical_features + categorical_features]
y_train_a = df_train['Winner_encoded']
model_a = make_pipeline()
model_a.fit(X_train_a, y_train_a)
raw_acc_a = accuracy_score(y_test, model_a.predict(X_test))
sym_acc_a, sym_brier_a = symmetrized_eval(model_a, X_test, y_test)
print(f"  Raw accuracy:  {raw_acc_a:.4f}")
print(f"  Sym accuracy:  {sym_acc_a:.4f}")
print(f"  Sym Brier:     {sym_brier_a:.4f}")

# --- Model B: With augmentation ---
print("\n=== Model B: WITH augmentation (corner-swap mirrors) ===")
df_mirror = df_train.copy()
df_mirror['Winner_encoded'] = 1 - df_mirror['Winner_encoded']
for feat in numerical_features:
    df_mirror[feat] = -df_mirror[feat]
df_train_b = pd.concat([df_train, df_mirror], ignore_index=True)
X_train_b = df_train_b[numerical_features + categorical_features]
y_train_b = df_train_b['Winner_encoded']
model_b = make_pipeline()
model_b.fit(X_train_b, y_train_b)
raw_acc_b = accuracy_score(y_test, model_b.predict(X_test))
sym_acc_b, sym_brier_b = symmetrized_eval(model_b, X_test, y_test)
print(f"  Raw accuracy:  {raw_acc_b:.4f}")
print(f"  Sym accuracy:  {sym_acc_b:.4f}")
print(f"  Sym Brier:     {sym_brier_b:.4f}")

# --- Model C: No augmentation, more features (include per-corner raw stats) ---
print("\n=== Model C: Per-corner features (NOT symmetric) ===")
per_corner_features = [
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
]
all_c_features = per_corner_features
num_t = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
prep_c = ColumnTransformer([('num', num_t, all_c_features)])
model_c = Pipeline([('preprocessor', prep_c),
                     ('classifier', GradientBoostingClassifier(
                         n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, min_samples_leaf=20, random_state=42))])

X_train_c = df_train[all_c_features]
X_test_c = df_test[all_c_features]
model_c.fit(X_train_c, y_train_a)
raw_acc_c = accuracy_score(y_test, model_c.predict(X_test_c))
print(f"  Raw accuracy:  {raw_acc_c:.4f}")

# Can we symmetrize per-corner? Yes: swap Red/Blue columns and average
X_test_c_rev = X_test_c.copy()
for col in all_c_features:
    if col.startswith('Red'):
        partner = col.replace('Red', 'Blue', 1)
        if partner in all_c_features:
            X_test_c_rev[col], X_test_c_rev[partner] = X_test_c[partner].values, X_test_c[col].values
fwd_c = model_c.predict_proba(X_test_c)
rev_c = model_c.predict_proba(X_test_c_rev)
sym_prob_c = (fwd_c[:, 1] + rev_c[:, 0]) / 2.0
sym_pred_c = (sym_prob_c > 0.5).astype(int)
sym_acc_c = accuracy_score(y_test, sym_pred_c)
sym_brier_c = brier_score_loss(y_test, sym_prob_c)
print(f"  Sym accuracy:  {sym_acc_c:.4f}")
print(f"  Sym Brier:     {sym_brier_c:.4f}")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  {'Model':<40s} {'Raw':>8s} {'Sym':>8s} {'Brier':>8s}")
print(f"  {'Baseline (always Red)':<40s} {baseline:>8.4f} {'N/A':>8s} {'N/A':>8s}")
print(f"  {'A: GBM, diffs, no augmentation':<40s} {raw_acc_a:>8.4f} {sym_acc_a:>8.4f} {sym_brier_a:>8.4f}")
print(f"  {'B: GBM, diffs, WITH augmentation':<40s} {raw_acc_b:>8.4f} {sym_acc_b:>8.4f} {sym_brier_b:>8.4f}")
print(f"  {'C: GBM, per-corner features':<40s} {raw_acc_c:>8.4f} {sym_acc_c:>8.4f} {sym_brier_c:>8.4f}")
