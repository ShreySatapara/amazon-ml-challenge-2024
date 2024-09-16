#!/bin/bash

python eval.py 
    --model_path <path/to/your/model>
    --test_data_path <path/to/test/data>
    --output_path <path/to/save predictions.csv> 
    --start_idx START_INDEX
    --end_idx END_INDEX
    --gpu_id GPU_ID