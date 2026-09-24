import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from sentence_transformers import SentenceTransformer
import torch

from .config import (
    TOP_K_TFIDF,
    TOP_K_EMB,
    MAX_CANDIDATES_PER_ENTITY,
    EMBEDDING_MODEL_NAME
)

class CandidateBlocker:
    def __init__(self, top_k_tfidf: int = TOP_K_TFIDF, top_k_emb: int = TOP_K_EMB, max_candidates: int = MAX_CANDIDATES_PER_ENTITY):
        self.top_k_tfidf = top_k_tfidf
        self.top_k_emb = top_k_emb
        self.max_candidates = max_candidates
        self.tfidf_vectorizer = None
        self.embedding_model = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading SentenceTransformer '{EMBEDDING_MODEL_NAME}' on {device}...")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        return self.embedding_model

    def run_blocking(
        self, 
        s1_df: pd.DataFrame, 
        s2s3_df: pd.DataFrame
    ) -> Tuple[Dict[str, List[Tuple[str, float, int, float, int]]], np.ndarray, np.ndarray]:
        """
        Runs TF-IDF and Dense Embedding multi-pass candidate blocking.
        
        Returns:
            candidates: Dict mapping s1_entity_id -> list of candidate tuples:
                (candidate_id, tfidf_score, tfidf_rank, emb_score, emb_rank)
            s1_emb: S1 dense embeddings
            s2s3_emb: S2/S3 dense embeddings
        """
        print("\n--- Starting Multi-Pass Blocking ---")
        s1_ids = s1_df['entity_id'].values
        s2s3_ids = s2s3_df['entity_id'].values
        
        s1_countries = s1_df['country_clean'].values
        s2s3_countries = s2s3_df['country_clean'].values
        
        # ----------------- PASS A: TF-IDF Char N-Grams -----------------
        print(f"Pass A: Building character n-gram TF-IDF on {len(s1_df) + len(s2s3_df)} total texts...")
        all_texts = list(s1_df['clean_combined'].values) + list(s2s3_df['clean_combined'].values)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=120_000,
            sublinear_tf=True
        )
        self.tfidf_vectorizer.fit(all_texts)
        
        s1_tfidf = self.tfidf_vectorizer.transform(s1_df['clean_combined'].values)
        s2s3_tfidf = self.tfidf_vectorizer.transform(s2s3_df['clean_combined'].values)
        
        print("Pass A: Computing TF-IDF sparse similarity matrix...")
        tfidf_sim = s1_tfidf.dot(s2s3_tfidf.T)
        
        # ----------------- PASS B: Sentence Embeddings + FAISS -----------------
        print(f"Pass B: Generating dense sentence embeddings...")
        model = self._get_embedding_model()
        
        s1_emb = model.encode(
            s1_df['clean_combined'].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        s2s3_emb = model.encode(
            s2s3_df['clean_combined'].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        print("Pass B: Running FAISS Inner Product search...")
        dim = s2s3_emb.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(s2s3_emb.astype(np.float32))
        
        emb_scores, emb_indices = faiss_index.search(s1_emb.astype(np.float32), self.top_k_emb)
        
        # ----------------- PASS C: Candidate Fusion & Soft Country Prioritization -----------------
        print("Pass C: Merging candidates with soft country prioritization...")
        candidates_dict: Dict[str, List[Tuple[str, float, int, float, int]]] = {}
        
        for i in range(len(s1_ids)):
            s1_id = s1_ids[i]
            s1_c = s1_countries[i]
            
            cand_info: Dict[str, Dict[str, any]] = {}
            
            # TF-IDF top-K
            row = tfidf_sim.getrow(i).toarray().flatten()
            if len(row) > 0:
                top_tfidf_idx = np.argsort(row)[-self.top_k_tfidf:][::-1]
                for rank, idx in enumerate(top_tfidf_idx):
                    score = float(row[idx])
                    if score > 0.05: # Minimal noise filter
                        cid = s2s3_ids[idx]
                        c_match = (s2s3_countries[idx] == s1_c) or (not s1_c) or (not s2s3_countries[idx])
                        cand_info[cid] = {
                            'tfidf_score': score,
                            'tfidf_rank': rank + 1,
                            'emb_score': 0.0,
                            'emb_rank': 999,
                            'country_match': c_match
                        }
            
            # Embedding top-K
            for rank in range(self.top_k_emb):
                idx = emb_indices[i, rank]
                score = float(emb_scores[i, rank])
                if idx >= 0:
                    cid = s2s3_ids[idx]
                    c_match = (s2s3_countries[idx] == s1_c) or (not s1_c) or (not s2s3_countries[idx])
                    if cid in cand_info:
                        cand_info[cid]['emb_score'] = score
                        cand_info[cid]['emb_rank'] = rank + 1
                    else:
                        cand_info[cid] = {
                            'tfidf_score': 0.0,
                            'tfidf_rank': 999,
                            'emb_score': score,
                            'emb_rank': rank + 1,
                            'country_match': c_match
                        }
                            
            # Sort by soft score (similarity + same-country priority boost)
            # High-similarity pairs make it even if country is noisy/mismatched
            sorted_cands = sorted(
                cand_info.items(),
                key=lambda x: ((x[1]['emb_score'] + x[1]['tfidf_score']) + (0.3 if x[1]['country_match'] else 0.0)),
                reverse=True
            )[:self.max_candidates]
            
            candidates_dict[s1_id] = [
                (cid, data['tfidf_score'], data['tfidf_rank'], data['emb_score'], data['emb_rank'])
                for cid, data in sorted_cands
            ]
            
        print(f"Blocking complete for {len(candidates_dict)} Source 1 entities.")
        return candidates_dict, s1_emb, s2s3_emb


def export_candidate_pairs(candidates_dict: Dict[str, List[Tuple[str, float, int, float, int]]], output_path: str):
    """Write candidate_pairs.tsv in required format: source1_entity_id\tcandidate_entity_ids"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidates_dict.items():
            cand_ids = [c[0] for c in cands]
            f.write(f"{s1_id}\t{','.join(cand_ids)}\n")
    print(f"Exported candidate pairs to {output_path}")
