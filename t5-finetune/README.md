# T5 Model Fine-tuning and Evaluation

This project contains scripts for fine-tuning a T5 model on a custom dataset and evaluating its performance.

## Files

1. `train.py`: Script for fine-tuning the T5 model.
2. `eval.py`: Script for evaluating the fine-tuned model and generating predictions.
3. `custom_dataset.py`: Contains the CustomDataset class for data preprocessing.

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- Pandas
- NumPy
- scikit-learn
- tqdm

You can install the required packages using:
```
pip install torch transformers pandas numpy scikit-learn tqdm
```


## Usage

### Training

To fine-tune the T5 model, run:

```
# For single GPU fine-tuning

CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_path <path_to_data> \
    --output_dir <path_to_output> \
    --model_name t5-large \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-5

# For multi-GPU fine-tuning 

accelerate launch --config_file accelerate_config.yaml \
    train.py \
    --data_path <path_to_data> \
    --output_dir <path_to_output> \
    --model_name t5-large \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-5
```
Arguments:
- `--data_path`: Path to the input CSV file containing the training data.
- `--output_dir`: Directory to save the fine-tuned models (default: "./weights").
- `--model_name`: Name of the pre-trained model to use (default: "t5-large").
- `--num_epochs`: Number of training epochs (default: 2).
- `--batch_size`: Training batch size (default: 8).
- `--learning_rate`: Learning rate for training (default: 5e-5).


This script will:
- Load and preprocess the data
- Initialize the T5 model and tokenizer
- Fine-tune the model
- Save the final and best models

### Evaluation

To evaluate the fine-tuned model and generate predictions, run:

```
python eval.py 
    --model_path <path/to/your/model>
    --test_data_path <path/to/test/data>
    --output_path <path/to/save predictions.csv> 
    [--start_idx START_INDEX] 
    [--end_idx END_INDEX] 
    [--gpu_id GPU_ID]
```

Arguments:
- `--model_path`: Path to the fine-tuned model (required)
- `--test_data_path`: Path to test data CSV file (required)
- `--output_path`: Path to save the predictions CSV (required)
- `--start_idx`: Start index for evaluation (optional)
- `--end_idx`: End index for evaluation (optional)
- `--gpu_id`: GPU ID to use for evaluation (optional, default is -1 for CPU)

This script will:
- Load the specified model and tokenizer
- Evaluate the model on the test dataset
- Save the predictions to a CSV file

## CustomDataset

The `CustomDataset` class in `custom_dataset.py` is responsible for preprocessing the data. It formats the input text based on the group ID, detected text, and entity name. For non-test data, it also prepares the target text.

## Notes

- The training script uses a seed for reproducibility.
- The evaluation script supports batch processing and can be run on specific data ranges using the `start_idx` and `end_idx` arguments.
- Make sure to adjust file paths in the scripts according to your directory structure.
