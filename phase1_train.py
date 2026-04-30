"""
Phase 1 Training - FAST VERSION (simplified features)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

print("=" * 70)
print("PHASE 1: CORE FEATURES MODEL TRAINING (FAST)")
print("=" * 70)

# Load ATP data 2024-2025 (recent, smaller, faster)
print("\n[1] Loading recent ATP data (2024-2025)...")
dfs = []
for year in [2024, 2025]:
    filepath = Path('data') / f'atp_matches_{year}.csv'
    if filepath.exists():
        df = pd.read_csv(filepath, low_memory=False)
        dfs.append(df)
        print(f"    ✓ {year}: {len(df)} matches")

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('tourney_date').reset_index(drop=True)
print(f"Total: {len(df):,} matches\n")

# Prepare
print("[2] Preparing data...")
df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce').fillna(999)
df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce').fillna(999)
df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
df = df.dropna(subset=['tourney_date'])
df = df.sort_values('tourney_date').reset_index(drop=True)
print(f"After cleaning: {len(df):,} matches\n")

# Build features (simple, no loops)
print("[3] Building Phase 1 features...")

# Feature 1: Rank difference
df['rank_diff'] = (df['loser_rank'] - df['winner_rank']).fillna(0)

# Feature 2: Surface encoding
surface_map = {'Hard': 1.0, 'Clay': 2.0, 'Grass': 3.0}
df['surface_code'] = df['surface'].map(surface_map).fillna(1.0)

# Feature 3: Best of
df['best_of_code'] = pd.to_numeric(df['best_of'], errors='coerce').fillna(3.0)

print(f"  rank_diff: mean={df['rank_diff'].mean():.2f}, std={df['rank_diff'].std():.2f}")
print(f"  surface_code: mean={df['surface_code'].mean():.2f}")
print(f"  best_of_code: mean={df['best_of_code'].mean():.2f}\n")

# Target: 1 if favorite (lower rank) won, 0 if upset
df['target'] = (df['winner_rank'] < df['loser_rank']).astype(int)
print(f"Target distribution:")
print(f"  Favorites won (1): {df['target'].sum():,} ({df['target'].mean()*100:.1f}%)")
print(f"  Upsets (0): {(1-df['target']).sum():,} ({(1-df['target']).mean()*100:.1f}%)\n")

# Train-test split (80-20 chronological)
print("[4] Train/test split...")
split_idx = int(len(df) * 0.8)
X_train = df[['rank_diff', 'surface_code', 'best_of_code']].iloc[:split_idx].fillna(0)
y_train = df['target'].iloc[:split_idx]
X_test = df[['rank_diff', 'surface_code', 'best_of_code']].iloc[split_idx:].fillna(0)
y_test = df['target'].iloc[split_idx:]

print(f"  Train: {len(X_train):,}")
print(f"  Test:  {len(X_test):,}\n")

# Scale and train
print("[5] Training RandomForest (100 trees, max_depth=10)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)
print("  ✓ Model trained\n")

# Evaluate
print("=" * 70)
print("PHASE 1 RESULTS")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_proba)

baseline_acc = max(y_test.mean(), 1 - y_test.mean())

print(f"\nBaseline Accuracy: {baseline_acc:.4f} (always predict majority class)")
print(f"\nModel Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
print(f"Improvement:       {(acc - baseline_acc)*100:+.2f}%")
print(f"\nPrecision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN (correct upsets predicted):    {cm[0][0]:,}")
print(f"  FP (upsets predicted as favs):    {cm[0][1]:,}")
print(f"  FN (favs predicted as upsets):    {cm[1][0]:,}")
print(f"  TP (correct favs predicted):      {cm[1][1]:,}")

# Feature importance
print(f"\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)
features = ['rank_diff', 'surface_code', 'best_of_code']
importances = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
for feat, imp in importances:
    bar_width = int(imp * 50)
    bar = "█" * bar_width
    print(f"  {feat:20s}: {imp:.4f} ({imp*100:5.1f}%) {bar}")

# Save
Path('models').mkdir(exist_ok=True)
with open('models/phase1_model.pkl', 'wb') as f:
    pickle.dump((model, scaler), f)
print(f"\n✓ Model saved to models/phase1_model.pkl")

print("=" * 70)
