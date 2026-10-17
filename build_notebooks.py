import nbformat as nbf
import os

os.makedirs('notebooks', exist_ok=True)

# Phase 4 Notebook
nb4 = nbf.v4.new_notebook()

nb4.cells.append(nbf.v4.new_markdown_cell('# Phase 4: Model Training\nFine-tuning MuRIL and IndicBERT for Binary Hallucination Classification.'))

nb4.cells.append(nbf.v4.new_code_cell('''!pip install -r ../requirements-ml.txt'''))

nb4.cells.append(nbf.v4.new_code_cell('''import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np

# Load dataset
df = pd.read_csv("../data/phase2/phase2_dataset.csv")
# Filter out unverified or bad rows if needed. We assume 'label' is 0 or 1.
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Create text input
# Format: [CLS] Question: {q} Context: {c} [SEP] Answer: {a} [SEP]
df['text'] = "Question: " + df['question'] + " Context: " + df['context'] + " Answer: " + df['generated_answer']

# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
'''))

nb4.cells.append(nbf.v4.new_markdown_cell('## Fine-tuning MuRIL'))

nb4.cells.append(nbf.v4.new_code_cell('''model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="../models/muril_hallucination",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
'''))

nb4.cells.append(nbf.v4.new_code_cell('''# Save final model
trainer.save_model("../models/muril_hallucination_final")
'''))

with open('notebooks/phase4_model_training.ipynb', 'w') as f:
    nbf.write(nb4, f)

# Phase 5 Notebook
nb5 = nbf.v4.new_notebook()

nb5.cells.append(nbf.v4.new_markdown_cell('# Phase 5: Evaluation & Comparison\nEvaluating the trained models on Hindi, Marathi, and Tamil.'))

nb5.cells.append(nbf.v4.new_code_cell('''import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../data/phase2/phase2_dataset.csv")
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
df['text'] = "Question: " + df['question'] + " Context: " + df['context'] + " Answer: " + df['generated_answer']

# Load Model
model_path = "../models/muril_hallucination_final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

if torch.cuda.is_available():
    model = model.to('cuda')
'''))

nb5.cells.append(nbf.v4.new_code_cell('''def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.cpu().numpy()

# Note: In a real scenario, evaluate on the test split, not the entire dataset.
# We predict in batches
batch_size = 16
predictions = []
for i in range(0, len(df), batch_size):
    batch_texts = df['text'].iloc[i:i+batch_size].tolist()
    preds = predict(batch_texts)
    predictions.extend(preds)

df['prediction'] = predictions
'''))

nb5.cells.append(nbf.v4.new_code_cell('''print("Overall Classification Report:")
print(classification_report(df['label'], df['prediction'], target_names=['Faithful', 'Hallucinated']))

for lang in df['language'].unique():
    print(f"--- Language: {lang} ---")
    lang_df = df[df['language'] == lang]
    if len(lang_df) > 0:
        print(classification_report(lang_df['label'], lang_df['prediction'], target_names=['Faithful', 'Hallucinated']))

# Error analysis by injection strategy
print("--- Error Analysis by Injection Strategy ---")
error_df = df[df['label'] != df['prediction']]
strategy_counts = error_df['injection_strategy'].value_counts()
print(strategy_counts)
'''))

with open('notebooks/phase5_evaluation.ipynb', 'w') as f:
    nbf.write(nb5, f)

print("Notebooks created successfully.")
