import os
import json
import csv
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Phase 3 Verification API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE2_DATASET = PROJECT_ROOT / "data" / "phase2" / "phase2_dataset.csv"
PHASE3_DIR = PROJECT_ROOT / "data" / "phase3"
VERIFIED_FILE = PHASE3_DIR / "human_verified.jsonl"

class VerificationRequest(BaseModel):
    sample_id: str
    label_faithfulness: str  # 'faithful', 'hallucinated', 'uncertain'
    comments: str = ""

def load_verified_ids():
    verified = set()
    if VERIFIED_FILE.exists():
        with open(VERIFIED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        verified.add(record.get("sample_id"))
                    except Exception:
                        pass
    return verified

@app.get("/api/samples")
def get_samples(limit: int = 10):
    if not PHASE2_DATASET.exists():
        # Mock data if the file doesn't exist yet for testing purposes
        return [{
            "sample_id": "mock-1",
            "context": "This is a mock context about the history of India.",
            "question": "What is the capital of India?",
            "reference_answer": "New Delhi",
            "generated_answer": "Mumbai is the capital of India.",
            "label": 1,
            "injection_strategy": "entity_substitution"
        }]
        
    try:
        verified_ids = load_verified_ids()
        samples = []
        
        with open(PHASE2_DATASET, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sample_id") not in verified_ids:
                    # Convert empty strings to None (similar to pd.isna)
                    for k, v in row.items():
                        if v == "":
                            row[k] = None
                    samples.append(row)
                    if len(samples) >= limit:
                        break
                        
        return samples
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify")
def verify_sample(req: VerificationRequest):
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the review
    review = req.model_dump()
    with open(VERIFIED_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(review, ensure_ascii=False) + "\n")
        
    return {"status": "success", "message": "Verification saved"}
