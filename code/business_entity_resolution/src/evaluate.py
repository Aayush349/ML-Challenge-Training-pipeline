import numpy as np
from typing import Dict, Set, List, Tuple
from .config import THRESHOLD_START, THRESHOLD_END, THRESHOLD_STEP

def compute_macro_f05(
    predictions_dict: Dict[str, Set[str]], 
    ground_truth_dict: Dict[str, Set[str]]
) -> Tuple[float, float, float]:
    """
    Computes official competition Macro F_0.5 score across all Source 1 entities.
    
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Singletons:
        - True empty, Pred empty -> 1.0
        - True empty, Pred non-empty -> 0.0 (penalized false merge)
        - True non-empty, Pred empty -> 0.0
    
    Returns:
        (macro_f05, macro_precision, macro_recall)
    """
    f05_list = []
    p_list = []
    r_list = []
    
    all_s1_ids = set(ground_truth_dict.keys())
    
    for s1_id in all_s1_ids:
        true_set = ground_truth_dict[s1_id]
        pred_set = predictions_dict.get(s1_id, set())
        
        # Singleton logic
        if len(true_set) == 0:
            if len(pred_set) == 0:
                f05_list.append(1.0)
                p_list.append(1.0)
                r_list.append(1.0)
            else:
                f05_list.append(0.0)
                p_list.append(0.0)
                r_list.append(1.0)
            continue
            
        if len(pred_set) == 0:
            f05_list.append(0.0)
            p_list.append(1.0)
            r_list.append(0.0)
            continue
            
        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)
        
        p_list.append(p)
        r_list.append(r)
        
        if (0.25 * p + r) == 0:
            f05_list.append(0.0)
        else:
            f05 = (1.25 * p * r) / (0.25 * p + r)
            f05_list.append(f05)
            
    return float(np.mean(f05_list)), float(np.mean(p_list)), float(np.mean(r_list))


def optimize_threshold(
    pair_scores: List[Tuple[str, str, float]],
    ground_truth_dict: Dict[str, Set[str]]
) -> Tuple[float, float]:
    """
    Sweeps thresholds to find the cutoff maximizing Macro F_0.5 on validation data.
    
    Args:
        pair_scores: List of tuples (s1_id, candidate_id, probability)
        ground_truth_dict: Dict of s1_id -> set of true matching IDs
    
    Returns:
        (best_threshold, best_f05)
    """
    # Group pairs by s1_id for rapid evaluation
    s1_candidates: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in ground_truth_dict.keys()}
    for s1_id, cid, prob in pair_scores:
        if s1_id in s1_candidates:
            s1_candidates[s1_id].append((cid, prob))
            
    best_thresh = 0.70
    best_score = -1.0
    
    print("\n--- Sweeping Thresholds for Macro F_0.5 Optimization ---")
    thresholds = np.arange(THRESHOLD_START, THRESHOLD_END + THRESHOLD_STEP / 2, THRESHOLD_STEP)
    
    for t in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1_id, cands in s1_candidates.items():
            matched = {cid for cid, prob in cands if prob >= t}
            preds[s1_id] = matched
            
        f05, p, r = compute_macro_f05(preds, ground_truth_dict)
        if f05 > best_score:
            best_score = f05
            best_thresh = float(t)
            
    print(f"Optimal Threshold: {best_thresh:.2f} -> Validation Macro F_0.5: {best_score:.4f}")
    return best_thresh, best_score
