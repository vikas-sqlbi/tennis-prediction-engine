#!/usr/bin/env python
"""Quick Phase 1 test"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("Loading 2024 ATP data...")
df = pd.read_csv('data/atp_matches_2024.csv')
print(f"Loaded {len(df)} matches")

# Prepare data
df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce').fillna(999)
df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce').fillna(999)

# Simple feature
df['rank_diff'] = df['loser_rank'] - df['winner_rank']

# Train-test split
split = int(0.8 * len(df))
X_train = df[['rank_diff']].iloc[:split].fillna(0)
X_test = df[['rank_diff']].iloc[split:].fillna(0)
y_train = np.ones(len(X_train))  # All winners
y_test = np.ones(len(X_test))

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=10, max_depth=5)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print("SUCCESS")
