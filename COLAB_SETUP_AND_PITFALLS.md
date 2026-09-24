# Google Colab Setup Guide & Common Pitfalls

This guide provides step-by-step instructions to run the **Business Entity Resolution** pipeline on **Google Colab** (using free T4 GPU for lightning-fast embeddings) along with critical pitfalls to avoid.

---

## Part 1: Step-by-Step Google Colab Execution

### Step 1: Open a New Colab Notebook and Enable GPU
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **Runtime** > **Change runtime type**.
3. Select **T4 GPU** (Hardware accelerator) > Click **Save**.

---

### Step 2: Upload Files or Mount Google Drive
You can either upload the project folder directly to the Colab files pane or mount your Google Drive:

```python
# Cell 1: Mount Google Drive (Recommended for persistent storage)
from google.colab import drive
drive.mount('/content/drive')

# Set working directory to where you placed the project
import os
os.chdir('/content/drive/MyDrive/ML_challenge_AWS') # Change to your drive path
!pwd
```

*Alternatively, if uploading a zip directly into `/content/`:*
```python
# Cell 1 (Alt): Unzip uploaded files directly in Colab
!unzip -q project_data.zip -d /content/
%cd /content/
```

Ensure your directory structure in Colab looks like this:
```
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       └── requirements.txt
├── utils/
│   └── validate_submission.py
└── output/
```

---

### Step 3: Install Dependencies
Run in a code cell:
```python
# Cell 2: Install required packages
!pip install -q rapidfuzz faiss-cpu sentence-transformers lightgbm
```

---

### Step 4: Run the Complete End-to-End Pipeline
Execute the pipeline:
```python
# Cell 3: Execute Training, Threshold Tuning, and Test Inference
!python code/business_entity_resolution/src/run_pipeline.py
```
*What this does:*
1. Loads and preprocesses `train_source1.tsv`, `train_source2.tsv`, and `train_source3.tsv`.
2. Runs TF-IDF and Sentence-Transformer (`all-MiniLM-L6-v2`) multi-pass candidate blocking.
3. Builds 24-dimensional feature representations for candidate pairs.
4. Trains LightGBM classifier with early stopping.
5. Sweeps probability thresholds on validation set to maximize **Macro $F_{0.5}$**.
6. Runs inference on the test set (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`).
7. Outputs `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

---

### Step 5: Validate Your Output Files
Before uploading anything to the portal, run the local validator script:
```python
# Cell 4: Validate submission compliance
!python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
You should see:
```
[PASS] Submission files strictly adhere to all challenge rules and schema!
```

---

### Step 6: Create Final Submission Zip Archive
Run this cell to package your submission correctly:
```python
# Cell 5: Package submission into required zip archive
!zip -r my_team_submission.zip \
  output/matching_results.tsv \
  output/candidate_pairs.tsv \
  code/business_entity_resolution/ \
  Documentation_template.md

from google.colab import files
files.download('output/matching_results.tsv') # Upload this to live leaderboard!
files.download('my_team_submission.zip')      # Keep for final code submission!
```

---

## Part 2: Critical Mistakes & Pitfalls to Avoid

### 1. Reading TSV Files Without `sep="\t"`
* **Pitfall**: Doing `pd.read_csv("file.tsv")` without `sep="\t"`.
* **Why it breaks**: Business addresses and ground truth columns contain commas. Default comma-separated reading turns the whole row into a single mangled column silently.
* **Fix**: Always specify `pd.read_csv(filepath, sep="\t")`.

---

### 2. Hardcoding Countries to `{US, India}`
* **Pitfall**: The training set contains only `US` and `India`, but the test set introduces **`France`**.
* **Why it breaks**: If you filter your dataframe by `df[df['country'].isin(['US', 'India'])]`, or use a one-hot encoder fitted only on train countries, your pipeline will either drop France entities or crash during test inference.
* **Fix**: Treat `country` as an open set of string labels. Our blocking code matches `s1_country == cand_country` dynamically, which seamlessly works for France without any hardcoded country lists.

---

### 3. Using the Default Classification Threshold (0.50)
* **Pitfall**: Using `prob >= 0.5` or `model.predict(X)` directly.
* **Why it breaks**: The challenge evaluates on **$F_{0.5}$**, where precision is weighted **$2\times$ heavier than recall**. At 0.50, you will get many false merges (false positives), which drastically degrades your score.
* **Fix**: Our pipeline automatically sweeps thresholds and sets a precision-biased threshold (typically between **0.70 and 0.85**).

---

### 4. False Merges on Singletons
* **Pitfall**: Forcing a match (e.g., assigning the top-1 candidate regardless of score).
* **Why it breaks**: A Source 1 entity with no true matches (singleton) scores **1.0** if you predict an empty string, but drops to **0.0** if you predict even one false candidate!
* **Fix**: If no candidate exceeds the high threshold, predict an empty list.

---

### 5. Violating the Subset Rule
* **Pitfall**: Predicting an entity in `matching_results.tsv` that was never generated in `candidate_pairs.tsv`.
* **Why it breaks**: The submission portal validator will flag this as a pipeline violation and disqualify the submission.
* **Fix**: The pipeline ensures `matching_results` is strictly filtered from the final candidates. Always run `validate_submission.py` to confirm.

---

### 6. Using External APIs or Lookups (Disqualification Risk)
* **Pitfall**: Querying Google Maps API, Nominatim, corporate registries, or web scraping.
* **Why it breaks**: This is **strictly prohibited** in the problem statement and leads to immediate disqualification.
* **Fix**: All models used here (`all-MiniLM-L6-v2`, `LightGBM`) run 100% offline and locally on the provided files.

---

### 7. Google Colab GPU / RAM Exhaustion
* **Pitfall**: Encoding all strings one-by-one or creating dense cross-product matrices in memory.
* **Why it breaks**: Comparing all $S_1 \times (S_2 \cup S_3)$ pairs directly would create tens of millions of rows, crashing Colab's 12GB RAM instantly.
* **Fix**: We use sparse TF-IDF matrix multiplications and FAISS GPU/CPU inner-product search with batch size 128, restricting candidate pairs to $\le 50$ per $S_1$ entity.
