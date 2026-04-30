"""
Direct execution of Phase 1 training
This file runs when imported as a test
"""

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path
    import sys
    
    print("=" * 70)
    print("PHASE 1: CORE FEATURES MODEL TRAINING")
    print("=" * 70)
    
    # Load all ATP data
    print("\n[1] Loading ATP data...")
    dfs = []
    for year in range(2021, 2027):
        filepath = Path('data') / f'atp_matches_{year}.csv'
        if filepath.exists():
            df = pd.read_csv(filepath, low_memory=False)
            dfs.append(df)
            print(f"    ✓ {year}: {len(df)} matches")
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('tourney_date').reset_index(drop=True)
    print(f"\nTotal: {len(df):,} matches")
    
    # Prepare data
    print("\n[2] Preparing data...")
    df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce').fillna(999)
    df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce').fillna(999)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['tourney_date', 'winner_name', 'loser_name'])
    df = df.sort_values('tourney_date').reset_index(drop=True)
    print(f"After cleaning: {len(df):,} matches\n")
    
    # Build Phase 1 features
    print("[3] Building Phase 1 features...")
    print("    - Feature 1: rank_diff (loser rank - winner rank)")
    print("    - Feature 2: surface_perf_diff (winner % - loser % on surface)")
    print("    - Feature 3: form_diff (winner last-10 % - loser last-10 %)\n")
    
    # Feature 1: Rank difference
    df['rank_diff'] = (df['loser_rank'] - df['winner_rank']).fillna(0)
    
    # Features 2 & 3: Cumulative stats (time-aware)
    player_wins = {}
    player_total = {}
    surface_wins = {}
    surface_total = {}
    player_last10 = {}
    
    surface_diffs = []
    form_diffs = []
    
    for idx, row in df.iterrows():
        winner = row['winner_name']
        loser = row['loser_name']
        surface = row.get('surface', 'Hard')
        
        # Abbreviate surface
        s_map = {'Hard': 'H', 'Clay': 'C', 'Grass': 'G'}
        s = s_map.get(surface, 'X')
        
        # Surface win % (BEFORE this match)
        w_surf_win = surface_wins.get((winner, s), 0)
        w_surf_total = surface_total.get((winner, s), 1)
        w_surf_pct = w_surf_win / w_surf_total
        
        l_surf_win = surface_wins.get((loser, s), 0)
        l_surf_total = surface_total.get((loser, s), 1)
        l_surf_pct = l_surf_win / l_surf_total
        
        surface_diff = w_surf_pct - l_surf_pct
        surface_diffs.append(surface_diff)
        
        # Form last-10 (BEFORE this match)
        w_last10 = player_last10.get(winner, [])
        l_last10 = player_last10.get(loser, [])
        
        w_form = np.mean(w_last10[-10:]) if w_last10 else 0.5
        l_form = np.mean(l_last10[-10:]) if l_last10 else 0.5
        
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
        
        if (idx + 1) % 5000 == 0:
            print(f"    Processed {idx+1:,} matches...")
    
    df['surface_perf_diff'] = surface_diffs
    df['form_diff'] = form_diffs
    print(f"    ✓ Features built\n")
    
    # Check features
    print("Feature statistics:")
    for col in ['rank_diff', 'surface_perf_diff', 'form_diff']:
        print(f"    {col:20s}: mean={df[col].mean():7.3f}, std={df[col].std():7.3f}")
    
    # Prepare target: 1 if favorite (lower rank) won, 0 if upset
    # In tennis, lower rank number = better rank = favorite
    df['target'] = (df['winner_rank'] < df['loser_rank']).astype(int)
    
    # Train-test split (80-20 chronological)
    print("\n[4] Train/test split (80-20 chronological)...")
    split_idx = int(len(df) * 0.8)
    X_train = df[['rank_diff', 'surface_perf_diff', 'form_diff']].iloc[:split_idx].fillna(0)
    y_train = df['target'].iloc[:split_idx]
    X_test = df[['rank_diff', 'surface_perf_diff', 'form_diff']].iloc[split_idx:].fillna(0)
    y_test = df['target'].iloc[split_idx:]
    
    print(f"    Train: {len(X_train):,}, Test: {len(X_test):,}\n")
    
    # Scale and train
    print("[5] Training RandomForest...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("    ✓ Model trained\n")
    
    # Evaluate
    print("=" * 70)
    print("RESULTS")
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
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:,}, FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}, TP={cm[1][1]:,}")
    
    # Feature importance
    print(f"\nFeature Importance:")
    features = ['rank_diff', 'surface_perf_diff', 'form_diff']
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True):
        print(f"    {feat:20s}: {imp:.4f} ({imp*100:5.1f}%)")
    
    # Save
    Path('models').mkdir(exist_ok=True)
    import pickle
    with open('models/phase1_model.pkl', 'wb') as f:
        pickle.dump((model, scaler), f)
    print(f"\n✓ Model saved to models/phase1_model.pkl")
    print("=" * 70)
