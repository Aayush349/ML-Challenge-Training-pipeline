import json
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Set, Tuple

from .config import (
    TEST_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    DEFAULT_THRESHOLD
)
from .preprocess import preprocess_dataframe
from .blocking import CandidateBlocker, export_candidate_pairs
from .features import build_feature_matrix

def load_test_data():
    """Load test sources."""
    print("Loading test data from TSVs...")
    s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t")
    s2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t")
    s3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t")
    return s1, s2, s3

def run_inference():
    """End-to-end inference producing candidate_pairs.tsv and matching_results.tsv."""
    s1, s2, s3 = load_test_data()
    
    print("\n--- Preprocessing Test Records (US, India, France) ---")
    s1_prep = preprocess_dataframe(s1)
    s2_prep = preprocess_dataframe(s2)
    s3_prep = preprocess_dataframe(s3)
    s2s3_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    
    # Valid set of IDs in test S2/S3 for integrity check
    valid_cand_ids = set(s2s3_prep['entity_id'].values)
    
    # 1. Candidate Blocking
    blocker = CandidateBlocker()
    candidates_dict, s1_emb, s2s3_emb = blocker.run_blocking(s1_prep, s2s3_prep)
    
    # Ensure every single test S1 entity is present in candidates_dict (even if 0 candidates)
    for s1_id in s1_prep['entity_id'].values:
        if s1_id not in candidates_dict:
            candidates_dict[s1_id] = []
            
    # 2. Export Candidate Pairs (REQUIRED output)
    export_candidate_pairs(candidates_dict, str(CANDIDATE_PAIRS_PATH))
    
    # 3. Build Features for Test Candidate Pairs
    X_test, _, pairs = build_feature_matrix(
        candidates_dict=candidates_dict,
        s1_df=s1_prep,
        s2s3_df=s2s3_prep,
        s1_emb=s1_emb,
        s2s3_emb=s2s3_emb,
        ground_truth=None
    )
    
    # 4. Load Trained Model & Threshold
    model_file = MODELS_DIR / "lgbm_model.txt"
    meta_file = MODELS_DIR / "pipeline_meta.json"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run training first!")
        
    model = lgb.Booster(model_file=str(model_file))
    
    threshold = DEFAULT_THRESHOLD
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            threshold = meta.get('optimal_threshold', DEFAULT_THRESHOLD)
    print(f"Loaded model. Using F0.5-optimized decision threshold: {threshold:.2f}")
    
    # 5. Predict Probabilities
    if len(X_test) > 0:
        probs = model.predict(X_test)
    else:
        probs = []
        
    # Group predictions by s1_id
    matches_by_s1: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_prep['entity_id'].values}
    
    for i, (s1_id, cid) in enumerate(pairs):
        if probs[i] >= threshold:
            if cid in valid_cand_ids and cid not in matches_by_s1[s1_id]:
                matches_by_s1[s1_id].append(cid)
                
    # 6. Export matching_results.tsv (REQUIRED leaderboard submission file)
    print(f"\nWriting final matches to {MATCHING_RESULTS_PATH}...")
    with open(MATCHING_RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        total_matched_entities = 0
        total_links = 0
        for s1_id in s1_prep['entity_id'].values:
            matched_list = matches_by_s1.get(s1_id, [])
            if matched_list:
                total_matched_entities += 1
                total_links += len(matched_list)
                f.write(f"{s1_id}\t{','.join(matched_list)}\n")
            else:
                # Singleton: empty string
                f.write(f"{s1_id}\t\n")
                
    total_s1 = len(s1_prep)
    print(f"\n=== Submission Summary ===")
    print(f"Total Source 1 entities: {total_s1}")
    print(f"Entities with predicted matches: {total_matched_entities} ({total_matched_entities/total_s1:.1%})")
    print(f"Entities identified as singletons: {total_s1 - total_matched_entities} ({(total_s1 - total_matched_entities)/total_s1:.1%})")
    print(f"Total links predicted: {total_links}")
    print(f"Outputs successfully generated at: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_inference()
