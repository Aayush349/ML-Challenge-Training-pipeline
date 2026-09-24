#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

def validate(matching_file: Path, candidate_file: Path, test_dir: Path):
    issues = []
    
    # 1. Check file existence
    if not matching_file.exists():
        issues.append(f"Missing matching results file: {matching_file}")
    if not candidate_file.exists():
        issues.append(f"Missing candidate pairs file: {candidate_file}")
    if not test_dir.exists():
        issues.append(f"Missing test directory: {test_dir}")
        
    if issues:
        for iss in issues:
            print(f"[FAIL] {iss}")
        return False
        
    # Read test source 1 entities
    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"
    
    expected_s1_ids = []
    with open(s1_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        idx = header.index('entity_id')
        for line in f:
            if line.strip():
                expected_s1_ids.append(line.split('\t')[idx].strip())
    expected_s1_set = set(expected_s1_ids)
    
    # Read valid S2 and S3 IDs
    valid_target_ids = set()
    for s_file in [s2_file, s3_file]:
        with open(s_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            idx = header.index('entity_id')
            for line in f:
                if line.strip():
                    valid_target_ids.add(line.split('\t')[idx].strip())
                    
    # Read candidate pairs
    cand_map = {}
    with open(candidate_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line_no, line in enumerate(f, start=2):
            parts = line.strip('\r\n').split('\t')
            if len(parts) < 1:
                continue
            s1_id = parts[0].strip()
            cands = [c.strip() for c in parts[1].split(',') if c.strip()] if len(parts) > 1 and parts[1].strip() else []
            if s1_id in cand_map:
                issues.append(f"Duplicate S1 ID in candidate_pairs.tsv at line {line_no}: {s1_id}")
            cand_map[s1_id] = set(cands)
            
    # Read matching results
    match_map = {}
    with open(matching_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line_no, line in enumerate(f, start=2):
            parts = line.strip('\r\n').split('\t')
            if len(parts) < 1:
                continue
            s1_id = parts[0].strip()
            matches = [m.strip() for m in parts[1].split(',') if m.strip()] if len(parts) > 1 and parts[1].strip() else []
            if s1_id in match_map:
                issues.append(f"Duplicate S1 ID in matching_results.tsv at line {line_no}: {s1_id}")
            
            # Check for duplicates within list
            if len(matches) != len(set(matches)):
                issues.append(f"Duplicate candidate IDs inside matching list for {s1_id}")
                
            # Check valid targets
            for mid in matches:
                if mid.startswith("S1-"):
                    issues.append(f"Self-match to Source 1 forbidden: {s1_id} -> {mid}")
                if mid not in valid_target_ids:
                    issues.append(f"Match ID {mid} for {s1_id} not found in test Source 2 or Source 3!")
                    
            # Check subset rule
            cand_set = cand_map.get(s1_id, set())
            for mid in matches:
                if mid not in cand_set:
                    issues.append(f"Subset violation: Match {mid} for {s1_id} was never in candidate_pairs.tsv!")
                    
            match_map[s1_id] = matches

    # Check coverage of Source 1
    missing_in_match = expected_s1_set - set(match_map.keys())
    if missing_in_match:
        issues.append(f"Missing {len(missing_in_match)} Source 1 entities in matching_results.tsv! Example: {list(missing_in_match)[:5]}")
        
    missing_in_cand = expected_s1_set - set(cand_map.keys())
    if missing_in_cand:
        issues.append(f"Missing {len(missing_in_cand)} Source 1 entities in candidate_pairs.tsv! Example: {list(missing_in_cand)[:5]}")

    if issues:
        print(f"[FAIL] Found {len(issues)} validation issues:")
        for idx, iss in enumerate(issues[:20], 1):
            print(f"  {idx}. {iss}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more.")
        return False
    else:
        print("[PASS] Submission files strictly adhere to all challenge rules and schema!")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    args = parser.parse_args()
    
    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
