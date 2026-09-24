import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
