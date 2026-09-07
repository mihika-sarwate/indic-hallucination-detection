import pandas as pd
import ast
import re
from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LANGUAGES

def normalize_text(text):
    """Basic text normalization: removing extra spaces."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

import numpy as np

def extract_answer(answer_col):
    """
    IndicQA 'answers' column is a dict/string representation of a dict with numpy arrays.
    Usually like: {'text': array(['answer text'], dtype=object), 'answer_start': array([123])}
    """
    if pd.isna(answer_col):
        return None
    try:
        # If it's saved as string representation of dict
        if isinstance(answer_col, str):
            answer_dict = eval(answer_col, {"array": np.array, "nan": np.nan, "object": object})
        else:
            answer_dict = answer_col
            
        if 'text' in answer_dict and len(answer_dict['text']) > 0:
            val = answer_dict['text'][0]
            if val and str(val).strip():
                return str(val)
    except Exception as e:
        pass
    return None

def process_language_data(lang):
    """
    Reads raw data for a language, normalizes, extracts answers, deduplicates, and saves.
    """
    print(f"Processing data for language: {lang}")
    processed_dfs = []
    
    # We will look for all splits (train, validation, test) for this language
    for split in ['train', 'validation', 'test']:
        raw_file = RAW_DATA_DIR / f"raw_{lang}_{split}.csv"
        if not raw_file.exists():
            continue
            
        print(f"  Reading {raw_file}")
        df = pd.read_csv(raw_file)
        
        # Keep necessary columns
        if 'context' not in df.columns or 'question' not in df.columns or 'answers' not in df.columns:
            print(f"  Warning: Missing required columns in {raw_file}")
            continue
            
        df['language'] = lang
        df['split'] = split
        
        # Extract answer
        df['reference_answer'] = df['answers'].apply(extract_answer)
        
        # Normalize
        df['context'] = df['context'].apply(normalize_text)
        df['question'] = df['question'].apply(normalize_text)
        df['reference_answer'] = df['reference_answer'].apply(normalize_text)
        
        # Drop rows with empty essential fields
        df = df.dropna(subset=['context', 'question', 'reference_answer'])
        df = df[df['context'] != ""]
        df = df[df['question'] != ""]
        df = df[df['reference_answer'] != ""]
        
        df = df[['language', 'split', 'context', 'question', 'reference_answer']]
        processed_dfs.append(df)
        
    if not processed_dfs:
        print(f"No valid data found for {lang}.")
        return None
        
    final_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Deduplicate
    initial_len = len(final_df)
    final_df = final_df.drop_duplicates(subset=['context', 'question'])
    print(f"  Deduplicated: {initial_len} -> {len(final_df)} records")
    
    # Save processed file
    output_path = PROCESSED_DATA_DIR / f"processed_{lang}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"  Saved processed data to {output_path}")
    
    return final_df

def process_all_datasets():
    print("Starting data processing...")
    for lang in LANGUAGES:
        process_language_data(lang)

if __name__ == "__main__":
    process_all_datasets()
