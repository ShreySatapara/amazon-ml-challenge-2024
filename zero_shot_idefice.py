import os
import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import pandas as pd
from tqdm import tqdm
def main(start_idx, end_idx, input_csv, output_csv):
    # Set the GPU ID for CUDA
    DEVICE = f"cuda:0"
    # Load model and processor
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        torch_dtype=torch.float16).to(DEVICE)

    # Load test data
    test_df = pd.read_csv(input_csv)

    # Ensure the end index does not exceed the length of the DataFrame
    end_idx = min(end_idx, len(test_df))

    # Placeholder for predictions
    predictions = []

    # Define entity_unit_map beforehand
    entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram','kilogram','microgram','milligram','ounce','pound','ton'},
    'maximum_weight_recommendation': {'gram','kilogram','microgram','milligram','ounce','pound','ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre','cubic foot','cubic inch','cup','decilitre','fluid ounce','gallon','imperial gallon','litre','microlitre','millilitre','pint','quart'}
}
    # Iterate through the rows between start_idx and end_idx
    for i, (idx, row) in tqdm(enumerate(test_df.iloc[start_idx:end_idx].iterrows()), 
                               total=end_idx - start_idx, desc="Processing"):
        # entity_name = row['entity_name']
        # image_link = row['image_link']
        message = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", 
                 "text": f"What is the {row['entity_name']} of the given product?",},
            ]}]
        # Load the image
        try:
            # print(row['image_link'])
            # print(f"./test_images/{row['image_link'].split('/')[-1]}")
            images = [load_image(f"../dataset/test_images/{row['image_link'].split('/')[-1]}")]
        
        # Process the prompt and images
            prompt = processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Generate prediction
            generated_ids = model.generate(**inputs, max_new_tokens=25)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            # Extract and clean prediction
            prediction = generated_texts[0].split("\nAssistant: ")[-1]
            # print(generated_texts[0].split("\nAssistant: ")[-1])
            # Save the index and prediction
            predictions.append({"index": idx, "prediction": prediction,"generated_texts":generated_texts[0]})
        except:
            predictions.append({"index": idx, "prediction": "","generated_texts":""})
        if (i + 1) % 1000 == 0:
            pred_df = pd.DataFrame(predictions)
            intermediate_output = f"{output_csv.split('.csv')[0]}_{start_idx + i + 1}.csv"
            pred_df.to_csv(intermediate_output, index=False)
            print(f"Intermediate predictions saved to {intermediate_output}")

    # Convert predictions to a DataFrame and save to the specified output CSV
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Process images and generate predictions.")
    parser.add_argument("--start", type=int, required=True, help="Start index for processing.")
    parser.add_argument("--end", type=int, required=True, help="End index for processing.")
    parser.add_argument("--input_csv", type=str, default="test.csv", help="Input CSV file (default is 'test.csv').")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV file (default is 'predictions.csv').")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args.start, args.end, args.input_csv, args.output_csv)
