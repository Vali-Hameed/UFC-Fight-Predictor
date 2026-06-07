"""Find the best model configuration for real-world UFC predictions."""
import pandas as pd
import numpy as np
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

# Engineered features
df['RedFinishWins'] = df['RedWinsByKO'].fillna(0) + df['RedWinsBySubmission'].fillna(0)
df['BlueFinishWins'] = df['BlueWinsByKO'].fillna(0) + df['BlueWinsBySubmission'].fillna(0)
df['RedTotalFights'] = df['RedWins'].fillna(df['RedFinishWins']) + df['RedLosses'].fillna(0) + df['RedDraws'].fillna(0)
df['BlueTotalFights'] = df['BlueWins'].fillna(df['BlueFinishWins']) + df['BlueLosses'].fillna(0) + df['BlueDraws'].fillna(0)
df['RedFinishRate'] = np.where(df['RedTotalFights'] > 0, df['RedFinishWins'] / df['RedTotalFights'], 0)
df['BlueFinishRate'] = np.where(df['BlueTotalFights'] > 0, df['BlueFinishWins'] / df['BlueTotalFights'], 0)

train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size].copy()
df_test = df.iloc[train_size:].copy()

# Per-corner features — the model learns the relationship between
# each fighter's individual stats and the outcome
per_corner_numerical = [
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
    'RedFinishRate', 'BlueFinishRate',
    'RedTotalFights', 'BlueTotalFights',
]

y_train = df_train['Winner_encoded']
y_test = df_test['Winner_encoded']
baseline = y_test.mean()

def symmetrized_eval(model, X_test, y_test, features):
    fwd = model.predict_proba(X_test)
    X_rev = X_test.copy()
    # Swap Red/Blue columns
    for col in features:
        if col.startswith('Red'):
            partner = col.replace('Red', 'Blue', 1)
            if partner in features:
                X_rev[col], X_rev[partner] = X_test[partner].values, X_test[col].values
    rev = model.predict_proba(X_rev)
    sym_prob = (fwd[:, 1] + rev[:, 0]) / 2.0
    sym_pred = (sym_prob > 0.5).astype(int)
    return accuracy_score(y_test, sym_pred), brier_score_loss(y_test, sym_prob)

def augment_per_corner(df_in, features):
    """Augment by swapping Red/Blue columns and flipping the label."""
    mirror = df_in.copy()
    mirror['Winner_encoded'] = 1 - mirror['Winner_encoded']
    for col in features:
        if col.startswith('Red'):
            partner = col.replace('Red', 'Blue', 1)
            if partner in features:
                mirror[col], mirror[partner] = df_in[partner].values, df_in[col].values
    return pd.concat([df_in, mirror], ignore_index=True)

print(f"Baseline (always Red): {baseline:.4f}")
print(f"Test: {len(y_test)} fights\n")

# --- Test configurations ---
configs = [
    ("GBM d=3 n=200 lr=0.05", dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=20)),
    ("GBM d=3 n=300 lr=0.05", dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=20)),
    ("GBM d=4 n=200 lr=0.05", dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=20)),
    ("GBM d=3 n=200 lr=0.1",  dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, min_samples_leaf=20)),
    ("GBM d=2 n=300 lr=0.05", dict(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=20)),
    ("GBM d=3 n=500 lr=0.03", dict(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.8, min_samples_leaf=20)),
]

print(f"{'Config':<30s} {'Augmented':<10s} {'Raw':>8s} {'Sym':>8s} {'Brier':>8s}")
print("-" * 70)

best_sym = 0
best_config = None

for name, params in configs:
    for aug in [False, True]:
        X_train = df_train[per_corner_numerical]
        X_test_data = df_test[per_corner_numerical]
        
        if aug:
            df_aug = augment_per_corner(df_train, per_corner_numerical)
            X_train = df_aug[per_corner_numerical]
            y_tr = df_aug['Winner_encoded']
        else:
            y_tr = y_train
        
        num_t = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        prep = ColumnTransformer([('num', num_t, per_corner_numerical)])
        model = Pipeline([('preprocessor', prep),
                          ('classifier', GradientBoostingClassifier(random_state=42, **params))])
        model.fit(X_train, y_tr)
        
        raw_acc = accuracy_score(y_test, model.predict(X_test_data))
        sym_acc, sym_brier = symmetrized_eval(model, X_test_data, y_test, per_corner_numerical)
        
        aug_label = "Yes" if aug else "No"
        print(f"  {name:<28s} {aug_label:<10s} {raw_acc:>8.4f} {sym_acc:>8.4f} {sym_brier:>8.4f}")
        
        if sym_acc > best_sym:
            best_sym = sym_acc
            best_config = (name, aug, params)

print(f"\nBest config: {best_config[0]}, augmented={best_config[1]}")
print(f"Best symmetrized accuracy: {best_sym:.4f}")
