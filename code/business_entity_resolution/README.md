# Business Entity Resolution Pipeline

This repository contains the end-to-end runnable machine learning pipeline for the **Business Entity Resolution Challenge**.

## Approach & Architecture
We utilize a **Hybrid GBM + Embeddings** framework:
1. **Multilingual Normalization**: Cleans business names, addresses, legal designations (Corp, Pvt Ltd, SARL, SAS, etc.), and extracts numerical components (PIN/ZIP codes, street digits).
2. **Multi-Pass Blocking**: Combines character 3-5 gram TF-IDF similarity with dense sentence embeddings (`all-MiniLM-L6-v2`, MIT license, ~22M parameters) indexed via FAISS, subject to country consistency. Outputs `output/candidate_pairs.tsv`.
3. **Discriminative Feature Extraction**: Generates lexical string similarities (Levenshtein, Jaro-Winkler, Token Sort/Set), numerical overlaps, dense embedding cosines, and retrieval rankings.
4. **LightGBM Classifier**: Trained on entity-disjoint validation splits with class weighting.
5. **Macro F0.5 Calibration & Singleton Preservation**: Sweeps probability thresholds specifically optimizing macro F0.5 (favoring precision 2x over recall) and preserving empty predictions for singletons. Outputs `output/matching_results.tsv`.

## Compliance
- **License**: MIT / Apache 2.0 across all dependencies and models.
- **Parameters**: ~22M parameters (`all-MiniLM-L6-v2`), well within the 8 Billion parameter ceiling.
- **External Data**: Strictly zero external lookup/APIs/geocoding used.

## Setup & Dependencies
Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run End-to-End
From this directory:

### 1. Execute Complete Pipeline (Train + Inference)
```bash
python src/run_pipeline.py
```

### 2. Run Stages Independently (Optional)
To train the model and calibrate the threshold:
```bash
python src/train.py
```
To run inference on the test dataset:
```bash
python src/inference.py
```

### 3. Validate Outputs
Run the validator from the project root:
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
