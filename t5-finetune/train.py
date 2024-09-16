import torch
import numpy as np
import pandas as pd
import random
import argparse
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from custom_dataset import CustomDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune T5 model for entity extraction"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./weights",
        help="Directory to save the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="t5-large",
        help="Name of the pre-trained model to use",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    return parser.parse_args()


args = parse_args()

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = T5ForConditionalGeneration.from_pretrained(args.model_name)

# Load and split data
data = pd.read_csv(args.data_path)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=seed)

# Create datasets
train_dataset = CustomDataset(tokenizer, train_data)
val_dataset = CustomDataset(tokenizer, val_data)


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    predictions = eval_pred.predictions[0].argmax(-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    cleaned_preds = [pred.strip() for pred in decoded_preds]
    cleaned_labels = [label.strip() for label in decoded_labels]

    exact_matches = sum(
        pred == label for pred, label in zip(cleaned_preds, cleaned_labels)
    )
    total = len(cleaned_preds)
    exact_match_score = exact_matches / total if total > 0 else 0

    return {"exact_match": exact_match_score}


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=4000,
    save_total_limit=3,
    learning_rate=args.learning_rate,
    lr_scheduler_type="cosine",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save the final model
final_model = trainer.model
final_model.save_pretrained(f"{args.output_dir}/fine_tuned_{args.model_name}_final")
tokenizer.save_pretrained(f"{args.output_dir}/fine_tuned_{args.model_name}_final")

# Load the best model
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    print(f"Loading best model from {best_model_path}")
    best_model = T5ForConditionalGeneration.from_pretrained(best_model_path)
else:
    print("No best model found, using final model")
    best_model = final_model

# Save the best model
best_model.save_pretrained(f"{args.output_dir}/fine_tuned_{args.model_name}_best")
tokenizer.save_pretrained(f"{args.output_dir}/fine_tuned_{args.model_name}_best")
