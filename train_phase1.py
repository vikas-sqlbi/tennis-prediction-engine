"""
Phase 1 Training - Core Features Model
Trains on ATP matches from data/ directory
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
print("PHASE 1: CORE FEATURES MODEL TRAINING")
print("=" * 70)

# Load all ATP data
print("\n[1] Loading ATP data from CSV files...")
start_time = time.time()
dfs = []
for year in range(2021, 2027):
    filepath = Path('data') / f'atp_matches_{year}.csv'
    if filepath.exists():
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"    ✓ Loaded {year}: {len(df)} matches")

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('tourney_date').reset_index(drop=True)
print(f"\nTotal loaded: {len(df):,} matches in {time.time()-start_time:.2f}s")

# Prepare data
print("\n[2] Preparing data...")
# Convert rank columns to numeric
df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce').fillna(999)
df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce').fillna(999)
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['tourney_date', 'winner_name', 'loser_name'])
df = df.sort_values('tourney_date').reset_index(drop=True)
print(f"After cleaning: {len(df):,} matches")

# Build Phase 1 features
print("\n[3] Building Phase 1 features (time-aware)...")
print(f"    - Feature 1: rank_diff = loser_rank - winner_rank")
print(f"    - Feature 2: surface_perf_diff = winner surface % - loser surface %")
print(f"    - Feature 3: form_diff = winner last-10 % - loser last-10 %")

start_feat = time.time()

# Feature 1: Simple rank difference
df['rank_diff'] = (df['loser_rank'] - df['winner_rank']).fillna(0)

# Features 2 & 3: Use cumulative approach (time-aware, no data leakage)
player_wins = {}
player_total = {}
surface_wins = {}
surface_total = {}
player_last10 = {}

rank_diffs = []
surface_diffs = []
form_diffs = []

for idx, row in df.iterrows():
    winner = row['winner_name']
    loser = row['loser_name']
    surface = row['surface']
    
    # Get stats BEFORE this match (no time travel)
    # Surface performance
    if surface == 'Hard':
        s = 'H'
    elif surface == 'Clay':
        s = 'C'
    elif surface == 'Grass':
        s = 'G'
    else:
        s = 'X'
    
    w_surf_win = surface_wins.get((winner, s), 0)
    w_surf_total = surface_total.get((winner, s), 1)
    w_surf_pct = w_surf_win / w_surf_total if w_surf_total > 0 else 0.5
    
    l_surf_win = surface_wins.get((loser, s), 0)
    l_surf_total = surface_total.get((loser, s), 1)
    l_surf_pct = l_surf_win / l_surf_total if l_surf_total > 0 else 0.5
    
    surface_diff = w_surf_pct - l_surf_pct
    surface_diffs.append(surface_diff)
    
    # Form (last 10 matches)
    w_last10 = player_last10.get(winner, [])
    l_last10 = player_last10.get(loser, [])
    
    w_form = sum(w_last10[-10:]) / len(w_last10[-10:]) if w_last10 else 0.5
    l_form = sum(l_last10[-10:]) / len(l_last10[-10:]) if l_last10 else 0.5
    
    form_diff = w_form - l_form
    form_diffs.append(form_diff)
    
    # Update stats AFTER calculating features
    player_wins[winner] = player_wins.get(winner, 0) + 1
    player_total[winner] = player_total.get(winner, 0) + 1
    player_total[loser] = player_total.get(loser, 0) + 1
    
    surface_wins[(winner, s)] = surface_wins.get((winner, s), 0) + 1
    surface_total[(winner, s)] = surface_total.get((winner, s), 0) + 1
    surface_total[(loser, s)] = surface_total.get((loser, s), 0) + 1
    
    if winner not in player_last10:
        player_last10[winner] = []
    if loser not in player_last10:
        player_last10[loser] = []
    
    player_last10[winner].append(1)
    player_last10[loser].append(0)
    
    # Print progress every 2000 matches
    if (idx + 1) % 2000 == 0:
        print(f"    Processed {idx+1:,} matches...")

df['surface_perf_diff'] = surface_diffs
df['form_diff'] = form_diffs

print(f"✓ Features built in {time.time()-start_feat:.2f}s")
print(f"  rank_diff: mean={df['rank_diff'].mean():.2f}, std={df['rank_diff'].std():.2f}")
print(f"  surface_perf_diff: mean={df['surface_perf_diff'].mean():.3f}, std={df['surface_perf_diff'].std():.3f}")
print(f"  form_diff: mean={df['form_diff'].mean():.3f}, std={df['form_diff'].std():.3f}")

# Create target variable
df['target'] = 1  # Winner (already in Winner column)

# Split data (80% train, 20% test - chronological)
print("\n[4] Train/test split...")
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

X_train = train_df[['rank_diff', 'surface_perf_diff', 'form_diff']].fillna(0)
y_train = train_df['target']
X_test = test_df[['rank_diff', 'surface_perf_diff', 'form_diff']].fillna(0)
y_test = test_df['target']

print(f"Train: {len(X_train)} matches")
print(f"Test: {len(X_test)} matches")

# Scale features
print("\n[5] Training model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
start_train = time.time()
model.fit(X_train_scaled, y_train)
print(f"✓ Model trained in {time.time()-start_train:.2f}s")

# Evaluate
print("\n[6] Evaluation Results")
print("-" * 70)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Feature importance
print(f"\n[7] Feature Importance")
print("-" * 70)
importance = model.feature_importances_
features = ['rank_diff', 'surface_perf_diff', 'form_diff']
for feat, imp in sorted(zip(features, importance), key=lambda x: x[1], reverse=True):
    print(f"  {feat:20s}: {imp:.4f} ({imp*100:.1f}%)")

# Save model
Path('models').mkdir(exist_ok=True)
with open('models/phase1_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)
print(f"\n✓ Model saved to models/phase1_model.pkl")

print("\n" + "=" * 70)
print("PHASE 1 COMPLETE")
print("=" * 70)
