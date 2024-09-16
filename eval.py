import os
import argparse
import torch
import csv
import argparse
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from datasets import load_dataset
from accelerate import PartialState
from tqdm import tqdm

# Function to set GPU device and process the data
def run_inference(start_idx, end_idx):
    
    # Define device map
    device_map = {"": PartialState().process_index}
    
    USE_LORA = False
    USE_QLORA = True
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False 
    )
    
    # Lora Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    
    # Quantization configuration
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    
    # Load model with optional quantization
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16,
        quantization_config=bnb_config if USE_QLORA else None,
    )
    
    # Load adapter and enable it
    model.load_adapter("./weights/checkpoint-320")
    model.enable_adapters()
    
    # Load evaluation dataset
    eval_dataset = load_dataset("csv", data_files="../dataset/test.csv")["train"]
    
    # Set batch size
    EVAL_BATCH_SIZE = 4
    
    # Store unique generated texts
    generated_texts_unique = []
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
    indices = []
    # Iterate over the evaluation dataset within the specified range
    for i in tqdm(range(start_idx, min(end_idx, len(eval_dataset)), EVAL_BATCH_SIZE)):
        examples = eval_dataset[i: i + EVAL_BATCH_SIZE]
        
        texts = []
        images = [[f'../dataset/test_images/{example.split("/")[-1]}'] for example in examples["image_link"]]
        
        queries = [f"Group id: {group_id}, What is the {entity_name.replace('item_','')} of this item? answer in one of the following units: {', '.join(entity_unit_map[entity_name])}"
                   for group_id, entity_name in zip(examples['group_id'], examples['entity_name'])]
        indices.extend(examples['index'])
        for q in queries:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": q}
                    ]
                }]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
        
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        
        generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        generated_texts_unique.extend(generated_texts)
    print(generated_texts_unique)
    # Save results to a CSV file
    output_file = f"./test_predictions/test_{start_idx}_{end_idx}.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Generated Text"])  # Write header
        
        # for ind, text in zip(indices, generated_texts_unique):
        #     writer.writerow([ind, text])
        
        for idx, text in enumerate(generated_texts_unique, start=start_idx):
            writer.writerow([idx, text])  # Write index and generated text
    
    print(f"Predictions saved to {output_file}")


# Main function with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference on a specified range of data with a specific GPU")
    
    # Add command-line arguments
    parser.add_argument('--start_idx', type=int, required=True, help="Start index for the dataset")
    parser.add_argument('--end_idx', type=int, required=True, help="End index for the dataset")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the run_inference function with parsed arguments
    run_inference(args.start_idx, args.end_idx)
