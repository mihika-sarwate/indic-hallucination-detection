import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset details
DATASET_NAME = "ai4bharat/IndicQA"
LANGUAGES = ["hi", "mr", "ta"]

# Column mapping (standardizing to our format)
COLUMN_MAPPING = {
    "context": "context",
    "question": "question",
    "answers": "reference_answer",  # Note: IndicQA 'answers' is a dict, we need to extract 'text'
}

# Validation constraints
MIN_CONTEXT_LENGTH = 10
MIN_QUESTION_LENGTH = 5
MIN_ANSWER_LENGTH = 1
