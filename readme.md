# Amazon ML Challegne 2024 
## Submission by Team DEFAULT
-   [Shreykumar Satapara](https://shreysatapara.github.io)
-   Sayanta Adhikari
-   Arkaprava Majumdar
-   Rishabh Karnad

## Preprocessing
- create a dataset folder which will contain train/test file and train test images
- download and store train test images at
-   - ```./dataset/train_images```
-   - ```./dataset/test_images```

#### Functions to download images are available in utils.py

### OCR using [easyOCR](https://github.com/JaidedAI/EasyOCR)

```
python easyocr_text_extract.py --start_idx <first row index> --end_idx <end_row_index>
```
EasyOCR use gpu to extract the text, set environment variable `CUDA_VISIBLE_DEVICES` if you wish to use some other gpu, default is 0.

Modify the image and input/output file path in python script as needed


## Zero Shot Inference using [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8bHuggingFaceM4/idefics2-8b)

```
python zero_shot_idefice.py --start <start_index > --end <end_index> --input_csv <input_csv_path> --output_csv <output_csv_path>
```

#### The default path to read images is set to `../dataset/test_images` modify it as needed.


## Finetuning Idefice2 with quantized LoRA
We have used hugginface transfomrers with accelerate for distributed training on 4 V100 GPUs. Modify no of gpus in `accelerate_config.yaml` as needed.

```
accelerate launch --config_file <config_file_path> train.py
```
#### there are default paths in train and data_collator for reading train.csv and train_images, change as per your directory structure.

### Inference

```
python eval.py --start_idx <start_idx> --end_idx <end_idx>
```

#### there are default paths for reading test.csv and test_images and to write the predictions.

## For T5 finetuning and evaluation take a look at ./T5_finetuning folder
