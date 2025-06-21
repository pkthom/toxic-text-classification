import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load CSV datasets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")

train_texts = train_df["text"].tolist()
train_labels = train_df["label"].tolist()  # Labels must be 0 or 1
val_texts = val_df["text"].tolist()
val_labels = val_df["label"].tolist()

# 2. Prepare tokenizer and model
model_name = "unitary/unbiased-toxic-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    ignore_mismatched_sizes=True
)
# Explicitly set problem type to single-label classification
model.config.problem_type = "single_label_classification"

# 3. Tokenize datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# 4. Define Dataset class
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Classification task requires labels as long (int64)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_encodings, train_labels)
val_dataset = ToxicDataset(val_encodings, val_labels)

# 5. Define evaluation metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no"
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 8. Train the model
trainer.train()

# 9. Evaluate the model
eval_results = trainer.evaluate()
print("Validation results:", eval_results)

# 10. Save the fine-tuned model and tokenizer
trainer.save_model("./finetuned-toxic-roberta")
tokenizer.save_pretrained("./finetuned-toxic-roberta")

