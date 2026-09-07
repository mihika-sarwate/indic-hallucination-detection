from .download import download_datasets
from .process import process_all_datasets

def run_pipeline():
    print("=== Phase 1: Dataset Pipeline Started ===")
    
    print("\n--- Step 1: Downloading Datasets ---")
    download_datasets()
    
    print("\n--- Step 2: Processing and Deduplicating Datasets ---")
    process_all_datasets()
    
    print("\n=== Phase 1: Dataset Pipeline Completed ===")

if __name__ == "__main__":
    run_pipeline()
