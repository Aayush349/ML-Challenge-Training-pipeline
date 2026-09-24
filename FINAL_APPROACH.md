# Business Entity Resolution — Final Approach Document

> **Strategy**: Hybrid GBM + Embeddings  
> **Target Metric**: Macro F₀.₅ (precision-heavy)  
> **Constraint**: Open-source model, MIT/Apache 2.0, ≤ 8B params

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 1 — Preprocessing & Normalization](#3-stage-1--preprocessing--normalization)
4. [Stage 2 — Blocking / Candidate Generation](#4-stage-2--blocking--candidate-generation)
5. [Stage 3 — Feature Engineering](#5-stage-3--feature-engineering)
6. [Stage 4 — Matching Model (LightGBM)](#6-stage-4--matching-model-lightgbm)
7. [Stage 5 — Threshold Calibration for F₀.₅](#7-stage-5--threshold-calibration-for-f05)
8. [Stage 6 — Singleton Handling](#8-stage-6--singleton-handling)
9. [Stage 7 — France Generalization Safety Net](#9-stage-7--france-generalization-safety-net)
10. [Validation & Evaluation Strategy](#10-validation--evaluation-strategy)
11. [Implementation Plan & File Structure](#11-implementation-plan--file-structure)
12. [Risk Analysis & Mitigation](#12-risk-analysis--mitigation)
13. [Dependency List](#13-dependency-list)

---

## 1. Executive Summary

We build a **two-stage pipeline**:

| Stage | What it does | Output |
|---|---|---|
| **Blocking** | For each S1 entity, retrieves top-K plausible candidates from S2 ∪ S3 using a union of TF-IDF character n-grams + sentence-embedding ANN search | `candidate_pairs.tsv` |
| **Matching** | Scores every candidate pair with a LightGBM classifier trained on hand-crafted string-similarity features + embedding cosine similarities, then applies a precision-biased threshold | `matching_results.tsv` |

### Why this strategy wins

| Alternative | Problem |
|---|---|
| Pure string rules (Jaccard/Levenshtein only) | Breaks on France — abbreviation & transliteration rules tuned on US/India won't transfer |
| Fine-tuned transformer (cross-encoder) | High risk: longer training, harder to debug, license/param constraints, marginal gain over well-featured GBM |
| Rule-based country branching | Violates the "don't hard-code" constraint; fails on unseen countries |
| **Hybrid GBM + embeddings (ours)** | Embeddings generalize across countries (language-agnostic similarity); GBM exploits precise string/numeric features; fast to build, iterate, debug |

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT DATA                                   │
│  train/test_source1.tsv  ·  train/test_source2.tsv  ·  *_source3   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  STAGE 1: PREPROCESS  │
                    │  Normalize names,     │
                    │  addresses, suffixes  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │  STAGE 2: BLOCKING (Candidate Gen) │
              │                                    │
              │  Pass A: TF-IDF char 3-5 grams     │
              │  Pass B: Sentence-Embedding + FAISS│
              │  Pass C: Country hard filter        │
              │  → Union top-K per S1 entity       │
              │                                    │
              │  OUTPUT: candidate_pairs.tsv        │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │  STAGE 3: FEATURE ENGINEERING      │
              │                                    │
              │  String sims (name & address)      │
              │  Embedding cosine sims             │
              │  Numeric / structural features     │
              │  Country match flag                │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │  STAGE 4: LIGHTGBM CLASSIFIER      │
              │  Binary: match / no-match          │
              │  P(match | S1, Sk) for every pair  │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │  STAGE 5: THRESHOLD + SINGLETONS   │
              │                                    │
              │  Sweep T ∈ [0.4, 0.95] on val set  │
              │  Maximize macro F₀.₅               │
              │  Empty list if all scores < T*     │
              │                                    │
              │  OUTPUT: matching_results.tsv       │
              └────────────────────────────────────┘
```

---

## 3. Stage 1 — Preprocessing & Normalization

### 3.1 Why it matters

Raw business data is *extremely* noisy. If `"Tata Consultancy Services Pvt. Ltd."` and `"TCS Private Limited"` aren't normalized before similarity computation, every downstream metric will underperform.

### 3.2 Name Normalization

```python
import re
import unicodedata

# 1. Unicode normalization — critical for French accents (é → e)
def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.lower().strip()

# 2. Legal suffix standardization (country-agnostic)
SUFFIX_MAP = {
    # English
    r'\bprivate\s+limited\b': 'pvtltd',
    r'\bpvt\.?\s*ltd\.?\b': 'pvtltd',
    r'\bp\.?\s*ltd\.?\b': 'pvtltd',
    r'\blimited\b': 'ltd',
    r'\bltd\.?\b': 'ltd',
    r'\bcorporation\b': 'corp',
    r'\bcorp\.?\b': 'corp',
    r'\bincorporated\b': 'inc',
    r'\binc\.?\b': 'inc',
    r'\bcompany\b': 'co',
    r'\bco\.?\b': 'co',
    r'\bl\.?l\.?c\.?\b': 'llc',
    r'\bl\.?l\.?p\.?\b': 'llp',
    # French (for test set generalization)
    r'\bsoci[eé]t[eé]\s+anonyme\b': 'sa',
    r'\bsoci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b': 'sarl',
    r'\bs\.?a\.?r\.?l\.?\b': 'sarl',
    r'\bs\.?a\.?s\.?u?\.?\b': 'sas',
    r'\be\.?u\.?r\.?l\.?\b': 'eurl',
    r'\bs\.?a\.?\b': 'sa',
}

# 3. Punctuation & connector normalization
def normalize_connectors(text: str) -> str:
    text = re.sub(r'\s*&\s*', ' and ', text)
    text = re.sub(r'\s*\+\s*', ' and ', text)
    text = re.sub(r"[''`]", '', text)          # remove apostrophes
    text = re.sub(r'[^\w\s]', ' ', text)       # remove remaining punctuation
    text = re.sub(r'\s+', ' ', text).strip()   # collapse whitespace
    return text
```

### 3.3 Address Normalization

```python
ADDRESS_ABBREVS = {
    r'\bstreet\b': 'st', r'\bst\.?\b': 'st',
    r'\broad\b': 'rd', r'\brd\.?\b': 'rd',
    r'\bavenue\b': 'ave', r'\bave\.?\b': 'ave',
    r'\bboulevard\b': 'blvd', r'\bblvd\.?\b': 'blvd',
    r'\bbuilding\b': 'bldg', r'\bbldg\.?\b': 'bldg',
    r'\bfloor\b': 'fl', r'\bfl\.?\b': 'fl',
    r'\bopposite\b': 'opp', r'\bopp\.?\b': 'opp',
    r'\bnear\b': 'nr', r'\bnr\.?\b': 'nr',
    r'\bdrive\b': 'dr', r'\bdr\.?\b': 'dr',
    r'\blane\b': 'ln', r'\bln\.?\b': 'ln',
    r'\bsuite\b': 'ste', r'\bste\.?\b': 'ste',
    # French
    r'\brue\b': 'rue',
    r'\bplace\b': 'pl',
    r'\bcedex\b': 'cedex',
}
```

### 3.4 Final Preprocessed Columns

For each record, produce:

| New Column | Content |
|---|---|
| `clean_name` | Normalized business name (Unicode + suffix + connector cleanup) |
| `clean_address` | Normalized address (abbreviations expanded, Unicode cleaned) |
| `clean_combined` | `clean_name + " " + clean_address` (used for embedding) |
| `name_tokens` | Sorted set of word tokens from `clean_name` |
| `address_tokens` | Sorted set of word tokens from `clean_address` |
| `address_numbers` | Set of all digit sequences extracted from address (PIN codes, street numbers) |
| `country_clean` | Lowercased, stripped country string |

> [!IMPORTANT]
> **Do NOT drop or filter records by country.** Every S1 entity (including France) must appear in your output. Country is used as a feature, not a filter.

---

## 4. Stage 2 — Blocking / Candidate Generation

### 4.1 Goal

Reduce the search space from O(|S1| × |S2∪S3|) ≈ millions of pairs to ~30–50 candidates per S1 entity, while keeping recall as high as possible (blocking recall is your **hard ceiling**).

### 4.2 Pass A: TF-IDF Character N-gram Retrieval

**Why**: Character n-grams are robust to typos, abbreviations, and word-order changes. This is the reliability backbone.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Build TF-IDF on clean_combined (name + address)
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=100_000,
    sublinear_tf=True,
)

# Fit on ALL records (S1 + S2 + S3) for consistent vocabulary
all_texts = list(s1_df['clean_combined']) + list(s2s3_df['clean_combined'])
vectorizer.fit(all_texts)

# Transform separately
s1_tfidf = vectorizer.transform(s1_df['clean_combined'])
s2s3_tfidf = vectorizer.transform(s2s3_df['clean_combined'])

# For each S1 entity, retrieve top-K by cosine similarity
# Use sparse matrix multiplication for efficiency
TOP_K_TFIDF = 25

sim_matrix = s1_tfidf.dot(s2s3_tfidf.T)  # sparse × sparse.T → sparse
for i in range(sim_matrix.shape[0]):
    row = sim_matrix.getrow(i).toarray().flatten()
    top_indices = np.argsort(row)[-TOP_K_TFIDF:][::-1]
    # store candidates...
```

### 4.3 Pass B: Sentence-Embedding ANN Search (FAISS)

**Why**: Embeddings capture semantic similarity that string metrics miss — e.g., `"Infosys Technologies"` ↔ `"Infosys Tech Solutions"`, or French name variants not covered by our abbreviation dictionary.

**Model Choice**: `sentence-transformers/all-MiniLM-L6-v2`
- ✅ MIT License
- ✅ ~22M parameters (well under 8B limit)
- ✅ Multilingual-friendly (handles French reasonably)
- ✅ Fast inference on CPU

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode all records
s1_embeddings = model.encode(s1_df['clean_combined'].tolist(),
                             batch_size=256, show_progress_bar=True,
                             normalize_embeddings=True)
s2s3_embeddings = model.encode(s2s3_df['clean_combined'].tolist(),
                               batch_size=256, show_progress_bar=True,
                               normalize_embeddings=True)

# Build FAISS index (Inner Product = cosine sim for normalized vectors)
dim = s2s3_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(s2s3_embeddings.astype('float32'))

# Search
TOP_K_EMB = 25
scores, indices = index.search(s1_embeddings.astype('float32'), TOP_K_EMB)
```

### 4.4 Pass C: Country Hard Filter

After retrieving candidates from Pass A and Pass B, **remove** any candidate whose `country_clean` does not match the S1 entity's `country_clean`. This is not hard-coding — it's a logical constraint (a business in India won't match one in the US).

> [!NOTE]
> This filter is applied as a **post-retrieval filter**, not a pre-index restriction. This way the FAISS index and TF-IDF index are built once on all S2∪S3 records, and the country filter just prunes the retrieved set.

### 4.5 Candidate Union & Output

```python
# For each S1 entity:
# 1. Union of candidates from Pass A and Pass B
# 2. Apply country filter
# 3. Cap at MAX_CANDIDATES = 50
# 4. Write to candidate_pairs.tsv

# candidate_pairs.tsv format:
# source1_entity_id\tcandidate_entity_ids
# S1-00001\tS2-00047,S2-00193,S3-00812,S3-00999
# S1-00002\tS3-00004
# S1-00003\t
```

### 4.6 Blocking Quality Metrics (validate on training set)

| Metric | Target | How to compute |
|---|---|---|
| **Blocking Recall** | ≥ 0.95 | % of true matches present in candidate set |
| **Reduction Ratio** | ≥ 0.99 | 1 − (# candidate pairs / # total possible pairs) |
| **Candidates per S1** | 20–50 avg | Mean size of candidate lists |

> [!WARNING]
> If blocking recall < 0.90, your pipeline has a hard recall ceiling. Increase `TOP_K_TFIDF` and `TOP_K_EMB`, or add a third blocking pass (e.g., phonetic Soundex blocking on name tokens).

---

## 5. Stage 3 — Feature Engineering

For every candidate pair `(S1_entity, S2/S3_candidate)`, compute the following feature vector (~20–25 features):

### 5.1 Name Similarity Features

| # | Feature | Library | Notes |
|---|---|---|---|
| 1 | `name_jaro_winkler` | `rapidfuzz` | Excellent for prefix-matching company names |
| 2 | `name_levenshtein_ratio` | `rapidfuzz` | Normalized edit distance |
| 3 | `name_token_sort_ratio` | `rapidfuzz` | Handles word-order transpositions |
| 4 | `name_token_set_ratio` | `rapidfuzz` | Handles subset/superset names |
| 5 | `name_partial_ratio` | `rapidfuzz` | Best substring match ratio |
| 6 | `name_jaccard_tokens` | custom | Jaccard on word token sets |
| 7 | `name_jaccard_3gram` | custom | Jaccard on character 3-gram sets |
| 8 | `name_common_token_count` | custom | # of shared word tokens |
| 9 | `name_common_token_frac` | custom | `common / max(len_a, len_b)` |
| 10 | `name_length_diff` | custom | `abs(len_a - len_b)` |
| 11 | `name_exact_match` | custom | Binary: are clean names identical? |

### 5.2 Address Similarity Features

| # | Feature | Notes |
|---|---|---|
| 12 | `addr_token_sort_ratio` | Word-order agnostic address similarity |
| 13 | `addr_jaccard_tokens` | Token-level overlap |
| 14 | `addr_number_jaccard` | **Critical**: Jaccard on extracted digit sequences (PIN/zip/street#) |
| 15 | `addr_number_overlap_count` | # of shared numbers |
| 16 | `addr_containment` | Is one address a substring of the other? |
| 17 | `addr_length_diff` | Address length difference |

### 5.3 Embedding Features

| # | Feature | Notes |
|---|---|---|
| 18 | `emb_cosine_combined` | Cosine similarity of `clean_combined` embeddings |
| 19 | `emb_cosine_name` | Cosine similarity of `clean_name` embeddings only |
| 20 | `emb_cosine_address` | Cosine similarity of `clean_address` embeddings only |

### 5.4 Structural / Metadata Features

| # | Feature | Notes |
|---|---|---|
| 21 | `same_country` | Binary: do countries match? (should always be 1 after blocking filter, but keep as safety) |
| 22 | `candidate_source` | Binary: is candidate from S2 (0) or S3 (1)? |
| 23 | `blocking_rank_tfidf` | Rank in TF-IDF retrieval (lower = more similar) |
| 24 | `blocking_score_tfidf` | Raw cosine score from TF-IDF retrieval |
| 25 | `blocking_rank_emb` | Rank in embedding retrieval |

### 5.5 Feature Engineering Code Skeleton

```python
from rapidfuzz import fuzz, distance
import numpy as np

def compute_features(s1_row, candidate_row, 
                     s1_emb, cand_emb,
                     s1_name_emb, cand_name_emb,
                     s1_addr_emb, cand_addr_emb,
                     blocking_rank_tfidf, blocking_score_tfidf,
                     blocking_rank_emb):
    
    n1 = s1_row['clean_name']
    n2 = candidate_row['clean_name']
    a1 = s1_row['clean_address']
    a2 = candidate_row['clean_address']
    
    nt1 = s1_row['name_tokens']
    nt2 = candidate_row['name_tokens']
    at1 = s1_row['address_tokens']
    at2 = candidate_row['address_tokens']
    an1 = s1_row['address_numbers']
    an2 = candidate_row['address_numbers']
    
    features = {}
    
    # --- Name features ---
    features['name_jaro_winkler'] = distance.JaroWinkler.similarity(n1, n2)
    features['name_levenshtein_ratio'] = fuzz.ratio(n1, n2) / 100.0
    features['name_token_sort_ratio'] = fuzz.token_sort_ratio(n1, n2) / 100.0
    features['name_token_set_ratio'] = fuzz.token_set_ratio(n1, n2) / 100.0
    features['name_partial_ratio'] = fuzz.partial_ratio(n1, n2) / 100.0
    
    name_inter = nt1 & nt2
    name_union = nt1 | nt2
    features['name_jaccard_tokens'] = len(name_inter) / max(len(name_union), 1)
    features['name_common_token_count'] = len(name_inter)
    features['name_common_token_frac'] = len(name_inter) / max(len(nt1), len(nt2), 1)
    features['name_length_diff'] = abs(len(n1) - len(n2))
    features['name_exact_match'] = int(n1 == n2)
    
    # Character n-gram Jaccard
    ng1 = set(n1[i:i+3] for i in range(len(n1)-2))
    ng2 = set(n2[i:i+3] for i in range(len(n2)-2))
    features['name_jaccard_3gram'] = len(ng1 & ng2) / max(len(ng1 | ng2), 1)
    
    # --- Address features ---
    features['addr_token_sort_ratio'] = fuzz.token_sort_ratio(a1, a2) / 100.0
    
    addr_inter = at1 & at2
    addr_union = at1 | at2
    features['addr_jaccard_tokens'] = len(addr_inter) / max(len(addr_union), 1)
    
    num_inter = an1 & an2
    num_union = an1 | an2
    features['addr_number_jaccard'] = len(num_inter) / max(len(num_union), 1)
    features['addr_number_overlap_count'] = len(num_inter)
    features['addr_containment'] = int(a1 in a2 or a2 in a1)
    features['addr_length_diff'] = abs(len(a1) - len(a2))
    
    # --- Embedding features ---
    features['emb_cosine_combined'] = float(np.dot(s1_emb, cand_emb))
    features['emb_cosine_name'] = float(np.dot(s1_name_emb, cand_name_emb))
    features['emb_cosine_address'] = float(np.dot(s1_addr_emb, cand_addr_emb))
    
    # --- Structural features ---
    features['same_country'] = int(
        s1_row['country_clean'] == candidate_row['country_clean']
    )
    features['candidate_source'] = int(
        candidate_row['entity_id'].startswith('S3')
    )
    features['blocking_rank_tfidf'] = blocking_rank_tfidf
    features['blocking_score_tfidf'] = blocking_score_tfidf
    features['blocking_rank_emb'] = blocking_rank_emb
    
    return features
```

---

## 6. Stage 4 — Matching Model (LightGBM)

### 6.1 Why LightGBM

| Factor | LightGBM | Cross-Encoder Transformer |
|---|---|---|
| Training time | Minutes | Hours |
| Debugging ease | High (feature importance, SHAP) | Low |
| License risk | MIT ✅ | Varies |
| Tabular + numeric features | Native strength | Requires tokenization hacks |
| Performance on ER tasks | Excellent with good features | Marginal improvement |

### 6.2 Training Data Preparation

```python
import pandas as pd
import lightgbm as lgb

# Positive pairs: from ground truth
# For each (S1_id, matched_id) in train_ground_truth.tsv → label = 1

# Negative pairs: from candidate set minus true matches
# For each S1_id, candidates NOT in ground truth → label = 0

# Negative sampling: use ALL negatives from candidate set (no down-sampling)
# This preserves the natural class imbalance the model will see at inference
```

### 6.3 Model Configuration

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'scale_pos_weight': neg_count / pos_count,  # handle class imbalance
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}

model = lgb.train(
    params,
    train_set=lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(50),
    ],
)
```

### 6.4 Feature Importance Analysis

After training, inspect:
```python
importance = model.feature_importance(importance_type='gain')
# Expected top features:
# 1. name_token_sort_ratio
# 2. emb_cosine_combined
# 3. name_jaro_winkler
# 4. addr_number_jaccard
# 5. name_jaccard_3gram
```

If any feature has near-zero importance, consider dropping it to reduce noise.

---

## 7. Stage 5 — Threshold Calibration for F₀.₅

### 7.1 Why this is critical

The default classification threshold (0.5) optimizes for **accuracy**, not for **F₀.₅**. Since F₀.₅ weights precision 2× over recall, the optimal threshold is almost always **higher** (0.70–0.85).

### 7.2 Threshold Sweep

```python
def compute_macro_f05(predictions_dict, ground_truth_dict):
    """
    predictions_dict: {s1_id: set of predicted match IDs}
    ground_truth_dict: {s1_id: set of true match IDs}
    """
    f05_scores = []
    
    for s1_id in ground_truth_dict:
        pred = predictions_dict.get(s1_id, set())
        true = ground_truth_dict[s1_id]
        
        if len(pred) == 0 and len(true) == 0:
            f05_scores.append(1.0)  # Correct singleton
            continue
        if len(pred) == 0 and len(true) > 0:
            f05_scores.append(0.0)  # Missed all matches
            continue
        if len(pred) > 0 and len(true) == 0:
            f05_scores.append(0.0)  # False merge on singleton
            continue
        
        tp = len(pred & true)
        precision = tp / len(pred)
        recall = tp / len(true)
        
        if precision + recall == 0:
            f05_scores.append(0.0)
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(f05)
    
    return np.mean(f05_scores)


# Sweep thresholds
best_threshold = 0.5
best_f05 = 0.0

for threshold in np.arange(0.40, 0.96, 0.01):
    preds = {}
    for s1_id, candidates_with_scores in val_scores.items():
        matched = {cid for cid, score in candidates_with_scores if score >= threshold}
        preds[s1_id] = matched
    
    f05 = compute_macro_f05(preds, val_ground_truth)
    
    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f}, Val F₀.₅: {best_f05:.4f}")
```

### 7.3 Expected Behavior

| Threshold range | Precision | Recall | F₀.₅ | Notes |
|---|---|---|---|---|
| 0.40–0.55 | Low-Medium | High | Suboptimal | Too many false merges |
| 0.55–0.70 | Medium | Medium-High | Good | Reasonable balance |
| **0.70–0.85** | **High** | **Medium** | **Best** | **Sweet spot for F₀.₅** |
| 0.85–0.95 | Very High | Low | Declining | Over-conservative, misses too many |

---

## 8. Stage 6 — Singleton Handling

### 8.1 The Singleton Trap

- A Source 1 entity with **no true matches** scores **1.0** if you predict empty, **0.0** if you predict even one match.
- Since singletons are included in the macro-average, a single false merge on a singleton hurts your overall score significantly.

### 8.2 Strategy

1. **Never force a match**. If all candidate scores < threshold, output an empty list.
2. **Optional: Singleton classifier**. Train a simple binary classifier: "does this S1 entity have any true matches at all?" based on features like:
   - Max candidate score from blocking
   - Max model probability across all candidates
   - Number of high-scoring candidates (score > 0.3)
   - If the classifier says "likely singleton", apply an even stricter threshold.
3. **Validation check**: On your held-out split, measure:
   - What % of true singletons get falsely matched?
   - Target: < 5% false merge rate on singletons.

---

## 9. Stage 7 — France Generalization Safety Net

### 9.1 The Risk

France doesn't appear in training. Any pipeline component that relies on country-specific patterns (US address formats, Indian PIN codes) may fail.

### 9.2 Mitigation Strategies

1. **Cross-country validation** (most important):
   ```
   Fold 1: Train on India → Validate on US (US is "unseen France")
   Fold 2: Train on US → Validate on India (India is "unseen France")
   ```
   If F₀.₅ drops drastically when the country is held out, your features are too country-specific.

2. **Feature audit**: Every feature must work on arbitrary string data:
   - ✅ Jaccard on tokens (language-agnostic)
   - ✅ Character n-gram similarity (works on any alphabet)
   - ✅ Embedding cosine similarity (multilingual model)
   - ✅ Digit/number overlap (universal)
   - ❌ Hardcoded PIN code regex (India-specific)
   - ❌ US state abbreviation lookup table

3. **Embedding model**: `all-MiniLM-L6-v2` handles French reasonably well. If you want better multilingual coverage, consider `paraphrase-multilingual-MiniLM-L12-v2` (Apache 2.0, ~118M params).

4. **French legal suffixes**: Already included in our preprocessing (SARL, SAS, SA, EURL, SASU).

---

## 10. Validation & Evaluation Strategy

### 10.1 Data Split

```python
from sklearn.model_selection import train_test_split

# Split by S1 entity IDs (entity-disjoint split)
s1_ids = train_ground_truth['source1_entity_id'].unique()
train_ids, val_ids = train_test_split(s1_ids, test_size=0.2, random_state=42)

# All ground truth rows for val_ids become validation set
# All ground truth rows for train_ids become training set
```

### 10.2 Validation Checklist

| Check | How | Target |
|---|---|---|
| Blocking recall | `true_matches_in_candidates / total_true_matches` | ≥ 0.95 |
| Blocking reduction ratio | `1 - (total_candidate_pairs / total_possible_pairs)` | ≥ 0.99 |
| Model AUC-ROC | Standard ROC on validation pairs | ≥ 0.95 |
| **Macro F₀.₅** | Per-entity F₀.₅, averaged | **Maximize** |
| Singleton accuracy | `correct_empty_predictions / total_true_singletons` | ≥ 0.95 |
| Cross-country F₀.₅ | Hold-out-country validation | Within 5% of full val F₀.₅ |

### 10.3 Pre-submission Validation

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Must print `PASS` before uploading.

---

## 11. Implementation Plan & File Structure

### 11.1 Directory Structure

```
<team_name>_submission/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocess.py          # Stage 1: Normalization
│       │   ├── blocking.py            # Stage 2: TF-IDF + Embedding blocking
│       │   ├── features.py            # Stage 3: Feature engineering
│       │   ├── train.py               # Stage 4: LightGBM training
│       │   ├── inference.py           # Stage 4+5: Score pairs → threshold → output
│       │   ├── evaluate.py            # Stage 5: Macro F₀.₅ evaluation & threshold sweep
│       │   ├── config.py              # Hyperparameters & paths
│       │   └── utils.py               # Shared utilities
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

### 11.2 Build Order & Time Estimates

| Priority | Task | Time | Impact |
|---|---|---|---|
| 1 | `preprocess.py` — Load & normalize data | 1–2 hrs | Foundation for everything |
| 2 | `blocking.py` — TF-IDF blocking only (Pass A) | 2–3 hrs | Gets you a baseline candidate set |
| 3 | `features.py` — String similarity features only | 2–3 hrs | Enough for a baseline model |
| 4 | `train.py` + `evaluate.py` — LightGBM + F₀.₅ sweep | 2–3 hrs | **First scoreable submission** |
| 5 | `blocking.py` — Add embedding blocking (Pass B) | 2–3 hrs | Lifts recall ceiling |
| 6 | `features.py` — Add embedding cosine features | 1–2 hrs | Significant model improvement |
| 7 | Threshold tuning & singleton optimization | 1–2 hrs | Final score push |
| 8 | Cross-country validation & France safety checks | 1–2 hrs | Robustness guarantee |

> [!TIP]
> **Get a submission on the leaderboard ASAP** (after Priority 4). Then iterate. A mediocre submission you can improve is infinitely better than a perfect pipeline that's never submitted.

---

## 12. Risk Analysis & Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Low blocking recall (missed true matches) | Medium | High — hard ceiling | Union multiple blocking methods; measure recall explicitly |
| France entities break the pipeline | Medium | High — entire country scores 0 | Cross-country validation; avoid hardcoded patterns |
| False merges on singletons | High | High — each costs 1.0 in the average | Conservative threshold; explicit singleton checking |
| Slow embedding inference | Low | Medium — delays iteration | Use `all-MiniLM-L6-v2` (smallest); batch encoding |
| Output format rejected by validator | Low | Critical — submission not scored | Run `validate_submission.py` before every upload |
| Overfitting LightGBM to training countries | Medium | Medium | Cross-country CV; regularization (`min_child_samples`, `feature_fraction`) |

---

## 13. Dependency List

```txt
# requirements.txt
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
rapidfuzz>=3.0
sentence-transformers>=2.2
faiss-cpu>=1.7
torch>=2.0
tqdm>=4.65
```

All libraries are MIT or Apache 2.0 licensed. Total model parameter count: ~22M (MiniLM) + LightGBM trees (negligible). Well within the 8B parameter constraint.

---

## Quick Reference: Decision Cheat Sheet

| Question | Answer |
|---|---|
| Which embedding model? | `all-MiniLM-L6-v2` (MIT, 22M params) — or `paraphrase-multilingual-MiniLM-L12-v2` if French accuracy matters more |
| Which classifier? | LightGBM binary classifier on ~25 features |
| How many candidates per S1? | 30–50 (union of top-25 TF-IDF + top-25 embedding, deduplicated, country-filtered) |
| What threshold? | Sweep [0.40, 0.95] on validation; expect optimal ~0.70–0.85 |
| What about singletons? | Empty list if all scores < threshold. Never force a match. |
| Train/val split? | 80/20 entity-disjoint split on S1 IDs |
| How to handle France? | Generic features + multilingual embeddings + cross-country validation |
