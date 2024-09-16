from torch.utils.data import Dataset
from .constants import entity_unit_map


class CustomDataset(Dataset):
    def __init__(self, tokenizer, data, max_len=256, test=False):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        group_id = row["group_id"]
        entityname = row["entity_name"]
        detected_text = row["detected_text_easyocr"]
        entity_value = entity_unit_map[entityname]

        input_text = f"Given Group-ID: {group_id} and context as {detected_text}. What is the {entityname} of the product? Use the units as {entity_value}."

        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        ).squeeze()

        attention_mask = input_ids != self.tokenizer.pad_token_id

        if not self.test:
            target_text = row["entity_value"]
            target_ids = self.tokenizer.encode(
                target_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
            ).squeeze()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": target_ids,
            }
        else:
            return {
                "index": row["index"],
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import pandas as pd

    data = pd.read_csv("data/test_sample.csv")
    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    dataset = CustomDataset(tokenizer, data)

    for sample in dataset:
        print(sample)
        break
