import sys
import time
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from .train import train_pipeline
    from .inference import run_inference
except (ImportError, ValueError):
    from train import train_pipeline
    from inference import run_inference

def main():
    print("=" * 60)
    print("  BUSINESS ENTITY RESOLUTION PIPELINE (HYBRID GBM + EMBEDDINGS)  ")
    print("=" * 60)
    start_time = time.time()
    
    # Step 1: Train & Optimize Threshold
    print("\n>>> STEP 1: MODEL TRAINING & F0.5 THRESHOLD CALIBRATION")
    train_pipeline()
    
    # Step 2: Inference & File Generation
    print("\n>>> STEP 2: INFERENCE & SUBMISSION GENERATION")
    run_inference()
    
    elapsed = time.time() - start_time
    print(f"\n[DONE] Pipeline completed in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    main()
