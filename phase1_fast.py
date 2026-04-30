"""
Ultra-fast Phase 1 training
Simplified features for quick execution
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import time

print("=" * 70)
print("PHASE 1: CORE FEATURES (FAST)")
print("=" * 70)

# Load only recent data (2024-2025) for speed
print("\n[1] Loading ATP data (2024-2025)...")
start_time = time.time()
dfs = []
for year in [2024, 2025]:
    filepath = Path('data') / f'atp_matches_{year}.csv'
    if filepath.exists():
        df = pd.read_csv(filepath, low_memory=False)
        dfs.append(df)
        print(f"    ✓ {year}: {len(df)} matches")

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('tourney_date').reset_index(drop=True)
print(f"\nTotal: {len(df):,} matches in {time.time()-start_time:.2f}s")

# Prepare
print("\n[2] Preparing data...")
df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce').fillna(999)
df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce').fillna(999)

# Feature 1: Rank difference
print("[3] Building features...")
df['rank_diff'] = (df['loser_rank'] - df['winner_rank']).fillna(0)

# Feature 2: Surface (simple)
surface_map = {'Hard': 1, 'Clay': 2, 'Grass': 3}
df['surface_code'] = df['surface'].map(surface_map).fillna(0)

# Feature 3: Best of (simple)
df['best_of'] = pd.to_numeric(df['best_of'], errors='coerce').fillna(3)

print(f"  Feature summary:")
print(f"    rank_diff: mean={df['rank_diff'].mean():.2f}")
print(f"    surface_code: unique={df['surface_code'].nunique()}")
print(f"    best_of: mean={df['best_of'].mean():.2f}")

# Train-test split
print(f"\n[4] Train/test split...")
split_idx = int(len(df) * 0.8)
X_train = df[['rank_diff', 'surface_code', 'best_of']].iloc[:split_idx].fillna(0)
y_train = np.ones(len(X_train))
X_test = df[['rank_diff', 'surface_code', 'best_of']].iloc[split_idx:].fillna(0)
y_test = np.ones(len(X_test))

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Scale
print(f"\n[5] Training...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
train_start = time.time()
model.fit(X_train_scaled, y_train)
train_time = time.time() - train_start
print(f"  Trained in {train_time:.2f}s")

# Evaluate
print("\n" + "=" * 70)
print("RESULTS (2024-2025 data)")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAccuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# Feature importance
print(f"\nFeature Importance:")
for feat, imp in zip(['rank_diff', 'surface_code', 'best_of'], model.feature_importances_):
    print(f"  {feat:20s}: {imp:.4f} ({imp*100:5.1f}%)")

# Save
Path('models').mkdir(exist_ok=True)
with open('models/phase1_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)
print(f"\n✓ Model saved")
print("=" * 70)
