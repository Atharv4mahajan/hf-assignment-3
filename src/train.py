import random
import json
import gzip
import requests
import os
from collections import defaultdict
from huggingface_hub import login
import os
# Login to HuggingFace
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)

import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ==============================
# CONFIG
# ==============================

model_name = "distilbert-base-cased"
max_length = 128
cached_model_directory_name = "distilbert-reviews-genres-small"

# Auto detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# DATA URLs
# ==============================

genre_url_dict = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}

# ==============================
# SMALL STREAMED LOADER
# ==============================

def load_reviews(url, head=500, sample_size=200):
    reviews = []
    count = 0

    response = requests.get(url, stream=True)

    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1

            if count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))

# ==============================
# LOAD DATA
# ==============================

train_texts = []
train_labels = []
test_texts = []
test_labels = []

print("Loading small dataset...")

for genre, url in genre_url_dict.items():
    print(f"Loading: {genre}")
    reviews = load_reviews(url, head=500, sample_size=200)

    reviews = random.sample(reviews, 200)

    for review in reviews[:150]:
        train_texts.append(review)
        train_labels.append(genre)

    for review in reviews[150:]:
        test_texts.append(review)
        test_labels.append(genre)

print(f"Train size: {len(train_texts)}")
print(f"Test size: {len(test_texts)}")

# ==============================
# LABEL ENCODING
# ==============================

unique_labels = sorted(list(set(train_labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

train_labels_encoded = [label2id[y] for y in train_labels]
test_labels_encoded = [label2id[y] for y in test_labels]

# ==============================
# TOKENIZER
# ==============================

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=max_length
)

test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, max_length=max_length
)

# ==============================
# DATASET CLASS
# ==============================

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, train_labels_encoded)
test_dataset = MyDataset(test_encodings, test_labels_encoded)

# ==============================
# MODEL
# ==============================

model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels)
).to(device)

# ==============================
# TRAINING ARGUMENTS
# ==============================

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    output_dir="./results",
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="no",
    report_to=[],
)

# ==============================
# METRICS
# ==============================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ==============================
# TRAINER
# ==============================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# ==============================
# TRAIN
# ==============================

print("Starting training...")
trainer.train()

# ==============================
# EVALUATE
# ==============================

print("Evaluating model...")
metrics = trainer.evaluate()
print(metrics)

# ==============================
# SAVE MODEL
# ==============================

trainer.save_model(cached_model_directory_name)
print("Model saved successfully!")
trainer.push_to_hub("hf-assignment-3-distilbert-genres")