import os
import pandas as pd
from datasets import load_dataset
from .config import DATASET_NAME, LANGUAGES, RAW_DATA_DIR

def download_datasets():
    """
    Downloads IndicQA for the specified languages and saves them as raw CSV files.
    """
    print(f"Downloading dataset {DATASET_NAME} for languages: {LANGUAGES}")
    
    for lang in LANGUAGES:
        print(f"Processing language: {lang}")
        try:
            # IndicQA datasets might have language-specific subsets
            # Adjust the subset name if IndicQA uses a different naming convention, like 'indicqa.hi'
            # ai4bharat/IndicQA subsets are like "hi", "mr", "ta"
            dataset = load_dataset(DATASET_NAME, f"indicqa.{lang}", trust_remote_code=True)
            
            for split in dataset.keys():
                df = dataset[split].to_pandas()
                output_path = RAW_DATA_DIR / f"raw_{lang}_{split}.csv"
                df.to_csv(output_path, index=False)
                print(f"Saved {lang} {split} split to {output_path} (Shape: {df.shape})")
                
        except Exception as e:
            print(f"Failed to download or save data for language {lang}: {e}")

if __name__ == "__main__":
    download_datasets()
