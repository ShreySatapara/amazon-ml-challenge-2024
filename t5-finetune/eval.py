import torch
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration
from custom_dataset import CustomDataset
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate T5 model and save predictions"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the predictions CSV",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to the test data CSV file",
    )
    parser.add_argument("--start_idx", type=int, required=False, help="Start index")
    parser.add_argument("--end_idx", type=int, required=False, help="End index")
    parser.add_argument("--gpu_id", type=int, required=False, help="GPU ID", default=-1)
    return parser.parse_args()


def load_model_and_tokenizer(model_path, device):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    return tokenizer, model


def load_and_prepare_data(tokenizer, test_data_path, start_idx, end_idx):
    test_data = pd.read_csv(test_data_path)
    if start_idx is not None and end_idx is not None:
        test_data = test_data.iloc[start_idx:end_idx]
    test_dataset = CustomDataset(tokenizer, test_data, test=True)
    return DataLoader(test_dataset, batch_size=128, shuffle=False)


def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    all_preds = []
    indices = []

    with torch.no_grad():
        for item in tqdm(dataloader, desc="Evaluating"):
            input_ids = item["input_ids"].to(device)
            attention_mask = item["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_length=128
            )
            pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(pred)
            indices.extend(item["index"].tolist())

    return indices, all_preds


def main():
    args = parse_arguments()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    tokenizer, model = load_model_and_tokenizer(args.model_path, device)
    test_dataloader = load_and_prepare_data(
        tokenizer, args.test_data_path, args.start_idx, args.end_idx
    )

    print("Starting evaluation...")
    indices, predictions = evaluate(model, test_dataloader, tokenizer, device)

    output_path = args.output_path.replace(
        ".csv", f"_{args.start_idx}_{args.end_idx}.csv"
    )
    df = pd.DataFrame({"index": indices, "prediction": predictions})
    df.to_csv(output_path, index=False)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
