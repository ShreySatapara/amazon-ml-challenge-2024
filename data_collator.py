import random


entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}


class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        self.flag = False  
        self.first_batch = None

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = f'../dataset/train_images/{example["image_link"].split("/")[-1]}'
            item_entity_name = example["entity_name"]
            if "item_" in item_entity_name:
                item_entity_name = item_entity_name.replace("item_", "")
            
            item_entity_group_id = example["group_id"]
            question = f"Group id: {item_entity_group_id}, What is the {item_entity_name} of this item? answer in one of the following units: {', '.join(entity_unit_map[example['entity_name']])}"
            answer = example["entity_value"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        try:
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == self.image_token_id] = -100
            batch["labels"] = labels
            if not self.flag:
                self.first_batch = batch
                self.flag = True
        except:
            print("Error in data collator")
            batch = self.first_batch
        return batch


