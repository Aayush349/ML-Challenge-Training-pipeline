import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from rapidfuzz import fuzz, distance

def compute_pair_features(
    s1_row: pd.Series,
    s2s3_row: pd.Series,
    s1_vec: np.ndarray,
    s2s3_vec: np.ndarray,
    tfidf_score: float,
    tfidf_rank: int,
    emb_score: float,
    emb_rank: int
) -> Dict[str, float]:
    """Computes a rich 24-dimensional feature vector for a candidate pair."""
    n1 = s1_row['clean_name']
    n2 = s2s3_row['clean_name']
    a1 = s1_row['clean_address']
    a2 = s2s3_row['clean_address']
    
    nt1 = s1_row['name_tokens']
    nt2 = s2s3_row['name_tokens']
    at1 = s1_row['address_tokens']
    at2 = s2s3_row['address_tokens']
    an1 = s1_row['address_numbers']
    an2 = s2s3_row['address_numbers']
    
    feat = {}
    
    # --- 1. Business Name Similarities ---
    feat['name_jaro_winkler'] = float(distance.JaroWinkler.similarity(n1, n2))
    feat['name_levenshtein_ratio'] = float(fuzz.ratio(n1, n2) / 100.0)
    feat['name_token_sort_ratio'] = float(fuzz.token_sort_ratio(n1, n2) / 100.0)
    feat['name_token_set_ratio'] = float(fuzz.token_set_ratio(n1, n2) / 100.0)
    feat['name_partial_ratio'] = float(fuzz.partial_ratio(n1, n2) / 100.0)
    
    name_inter = nt1 & nt2
    name_union = nt1 | nt2
    feat['name_jaccard_tokens'] = float(len(name_inter) / max(len(name_union), 1))
    feat['name_common_tokens'] = float(len(name_inter))
    feat['name_common_token_frac'] = float(len(name_inter) / max(len(nt1), len(nt2), 1))
    feat['name_len_diff'] = float(abs(len(n1) - len(n2)))
    feat['name_exact_match'] = float(1.0 if n1 == n2 and len(n1) > 0 else 0.0)
    
    # Char 3-gram Jaccard on name
    ng1 = {n1[i:i+3] for i in range(len(n1)-2)} if len(n1) >= 3 else set()
    ng2 = {n2[i:i+3] for i in range(len(n2)-2)} if len(n2) >= 3 else set()
    feat['name_jaccard_3gram'] = float(len(ng1 & ng2) / max(len(ng1 | ng2), 1))
    
    # --- 2. Address & Numeric Similarities ---
    feat['addr_token_sort_ratio'] = float(fuzz.token_sort_ratio(a1, a2) / 100.0)
    feat['addr_partial_ratio'] = float(fuzz.partial_ratio(a1, a2) / 100.0)
    
    addr_inter = at1 & at2
    addr_union = at1 | at2
    feat['addr_jaccard_tokens'] = float(len(addr_inter) / max(len(addr_union), 1))
    
    # Numbers/PIN/Street comparison - High discriminative power
    num_inter = an1 & an2
    num_union = an1 | an2
    feat['addr_num_jaccard'] = float(len(num_inter) / max(len(num_union), 1))
    feat['addr_num_overlap_count'] = float(len(num_inter))
    feat['addr_containment'] = float(1.0 if (a1 in a2 or a2 in a1) and len(a1) > 0 and len(a2) > 0 else 0.0)
    feat['addr_len_diff'] = float(abs(len(a1) - len(a2)))
    
    # --- 3. Dense Embedding & Cosine Similarity ---
    feat['emb_cosine'] = float(np.dot(s1_vec, s2s3_vec)) if s1_vec is not None and s2s3_vec is not None else emb_score
    
    # --- 4. Structural & Retrieval Signals ---
    feat['tfidf_score'] = float(tfidf_score)
    feat['tfidf_rank'] = float(tfidf_rank)
    feat['emb_score'] = float(emb_score)
    feat['emb_rank'] = float(emb_rank)
    
    # Explicit single-pass and multi-pass indicators
    feat['found_by_tfidf'] = float(1.0 if tfidf_rank < 999 else 0.0)
    feat['found_by_emb'] = float(1.0 if emb_rank < 999 else 0.0)
    feat['found_by_both'] = float(1.0 if (tfidf_rank < 999 and emb_rank < 999) else 0.0)
    
    feat['same_country'] = float(1.0 if s1_row['country_clean'] == s2s3_row['country_clean'] else 0.0)
    feat['is_source3'] = float(1.0 if s2s3_row['entity_id'].startswith('S3-') else 0.0)
    
    return feat

def build_feature_matrix(
    candidates_dict: Dict[str, List[Tuple[str, float, int, float, int]]],
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    s1_emb: np.ndarray = None,
    s2s3_emb: np.ndarray = None,
    ground_truth: Dict[str, Set[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds the feature DataFrame X, binary label vector y (if ground truth provided),
    and pair identifier list (s1_id, candidate_id).
    """
    s1_map = s1_df.set_index('entity_id')
    s2s3_map = s2s3_df.set_index('entity_id')
    
    s1_idx_map = {eid: idx for idx, eid in enumerate(s1_df['entity_id'].values)}
    s2s3_idx_map = {eid: idx for idx, eid in enumerate(s2s3_df['entity_id'].values)}
    
    rows = []
    labels = []
    pairs = []
    
    print("\nExtracting feature vectors for all candidate pairs...")
    for s1_id, cands in candidates_dict.items():
        if s1_id not in s1_map.index:
            continue
        s1_row = s1_map.loc[s1_id]
        s1_vec = s1_emb[s1_idx_map[s1_id]] if s1_emb is not None else None
        
        true_matches = ground_truth.get(s1_id, set()) if ground_truth is not None else None
        
        for cid, tf_score, tf_rank, e_score, e_rank in cands:
            if cid not in s2s3_map.index:
                continue
            s2s3_row = s2s3_map.loc[cid]
            s2s3_vec = s2s3_emb[s2s3_idx_map[cid]] if s2s3_emb is not None else None
            
            feat = compute_pair_features(
                s1_row, s2s3_row, s1_vec, s2s3_vec,
                tf_score, tf_rank, e_score, e_rank
            )
            rows.append(feat)
            pairs.append((s1_id, cid))
            
            if true_matches is not None:
                labels.append(1 if cid in true_matches else 0)
                
    X = pd.DataFrame(rows)
    y = np.array(labels) if ground_truth is not None else None
    print(f"Feature matrix built: {X.shape[0]} candidate pairs, {X.shape[1]} features.")
    if y is not None:
        pos_cnt = int(np.sum(y))
        print(f"Label distribution: {pos_cnt} positives, {len(y) - pos_cnt} negatives (Pos ratio: {pos_cnt/len(y):.3%})")
        
    return X, y, pairs
