import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import Dict, Set

from .config import (
    TRAIN_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    VAL_SIZE,
    MAX_BOOST_ROUNDS,
    EARLY_STOPPING_ROUNDS
)
from .preprocess import preprocess_dataframe
from .blocking import CandidateBlocker
from .features import build_feature_matrix
from .evaluate import compute_macro_f05, optimize_threshold

def load_train_data():
    """Load training sources and ground truth."""
    print("Loading training data from TSVs...")
    s1 = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t")
    s2 = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t")
    s3 = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t")
    gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t")
    
    # Parse ground truth to dictionary
    gt_dict: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1_id = str(row['source1_entity_id']).strip()
        matched = str(row['matched_entity_ids']).strip() if pd.notna(row['matched_entity_ids']) else ""
        if matched:
            gt_dict[s1_id] = set(m.strip() for m in matched.split(',') if m.strip())
        else:
            gt_dict[s1_id] = set()
            
    return s1, s2, s3, gt_dict

def train_pipeline():
    """Executes the complete model training and threshold calibration workflow."""
    s1, s2, s3, gt_dict = load_train_data()
    
    print("\n--- Preprocessing Raw Records ---")
    s1_prep = preprocess_dataframe(s1)
    s2_prep = preprocess_dataframe(s2)
    s3_prep = preprocess_dataframe(s3)
    s2s3_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    
    # 1. Entity-disjoint split on Source 1 IDs
    all_s1_ids = s1_prep['entity_id'].values
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )
    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)
    print(f"Disjoint Split: {len(train_s1_set)} Train S1 entities, {len(val_s1_set)} Validation S1 entities.")
    
    # 2. Run Candidate Blocking
    blocker = CandidateBlocker()
    candidates_dict, s1_emb, s2s3_emb = blocker.run_blocking(s1_prep, s2s3_prep)
    
    # Verify blocking recall on full ground truth
    total_true_matches = sum(len(matches) for matches in gt_dict.values())
    recalled_matches = 0
    for s1_id, matches in gt_dict.items():
        if s1_id in candidates_dict:
            cand_set = {c[0] for c in candidates_dict[s1_id]}
            recalled_matches += len(matches & cand_set)
    blocking_recall = recalled_matches / max(total_true_matches, 1)
    print(f"\n[Blocking Quality Check] Recall Ceiling: {blocking_recall:.4%} ({recalled_matches}/{total_true_matches} true pairs captured)")
    
    # 3. Build Feature Matrix
    X, y, pairs = build_feature_matrix(
        candidates_dict=candidates_dict,
        s1_df=s1_prep,
        s2s3_df=s2s3_prep,
        s1_emb=s1_emb,
        s2s3_emb=s2s3_emb,
        ground_truth=gt_dict
    )
    
    # 4. Partition pairs into Train vs Validation based on S1 ID
    train_mask = np.array([p[0] in train_s1_set for p in pairs])
    val_mask = np.array([p[0] in val_s1_set for p in pairs])
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs = [p for i, p in enumerate(pairs) if val_mask[i]]
    
    print(f"Train candidate pairs: {len(X_train)} (Pos: {np.sum(y_train)})")
    print(f"Val candidate pairs: {len(X_val)} (Pos: {np.sum(y_val)})")
    
    # 5. Train LightGBM Model
    # We train on the natural class distribution without scale_pos_weight distortion,
    # and use average_precision (PR-AUC) as early stopping metric to align with precision-heavy F0.5
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': -1,
        'seed': RANDOM_STATE
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    print("\nTraining LightGBM Classifier (Early stopping on PR-AUC)...")
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'val'],
        num_boost_round=MAX_BOOST_ROUNDS,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # 6. Evaluate Feature Importance
    print("\n--- Top Features by Gain ---")
    importance = model.feature_importance(importance_type='gain')
    feature_names = X.columns
    sorted_feat = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for feat_name, imp in sorted_feat[:10]:
        print(f"  {feat_name:25s}: {imp:.2f}")
        
    # 7. Optimize Threshold for Macro F_0.5 on Validation Set
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    val_pair_scores = [
        (val_pairs[i][0], val_pairs[i][1], float(val_probs[i]))
        for i in range(len(val_pairs))
    ]
    val_gt_dict = {s1_id: gt_dict[s1_id] for s1_id in val_s1_set}
    
    best_thresh, best_f05 = optimize_threshold(val_pair_scores, val_gt_dict)
    
    # 8. Save Model and Metadata
    model_file = MODELS_DIR / "lgbm_model.txt"
    model.save_model(str(model_file))
    print(f"\nModel saved to {model_file}")
    
    meta_file = MODELS_DIR / "pipeline_meta.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({
            'optimal_threshold': best_thresh,
            'val_macro_f05': best_f05,
            'features': list(X.columns)
        }, f, indent=2)
    print(f"Metadata and threshold saved to {meta_file}")
    
    return model, best_thresh

if __name__ == "__main__":
    train_pipeline()
