import os
import argparse
import pandas as pd
import easyocr
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='OCR text extraction using EasyOCR on a single GPU.')
parser.add_argument('--start_idx', type=int, required=True, help='Start index in the CSV for processing')
parser.add_argument('--end_idx', type=int, required=True, help='End index in the CSV for processing')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # Set the GPU ID

# Initialize EasyOCR reader on the specified GPU
reader = easyocr.Reader(['en'], gpu=f"cuda:0")

# Load the dataframe
df = pd.read_csv('../dataset/test.csv')  # Adjust the path to your dataframe

# Define the folder containing the images
image_folder = '../dataset/test_images/'
tqdm.pandas()
# Function to process an image and return detected text
def extract_text(row):
    image_link = row.image_link
    image_name = image_link.split("/")[-1]
    image_path = os.path.join(image_folder, image_name)
    
    # Run EasyOCR on the image if it exists
    if os.path.exists(image_path):
        try:
            result = reader.readtext(image_path)
            detected_text = " ".join([text for (bbox, text, prob) in result])
            return detected_text
        except Exception as e:
            print(f"Error processing {image_name} on GPU {args.gpu_id}: {e}")
            return None
    else:
        print(f"Image {image_name} not found")
        return None

# Subset the dataframe based on the provided start and end index
df_subset = df.iloc[args.start_idx:args.end_idx]

# Apply the OCR function to the subset of the dataframe use tqdm for progress bar
df_subset['detected_text_easyocr'] = df_subset.progress_apply(extract_text, axis=1)

# Save the updated subset dataframe to a new CSV
output_file = f'../dataset/test_ocr_{args.start_idx}_{args.end_idx}.csv'
df_subset.to_csv(output_file, index=False)

print(f"Text extraction completed for rows {args.start_idx} to {args.end_idx} using GPU {args.gpu_id}.")