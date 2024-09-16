import os
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from data_collator import MyDataCollator
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

from accelerate import PartialState
device_map={"": PartialState().process_index}

USE_LORA = False
USE_QLORA = True


processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False 
)


# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning

lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    use_dora=False if USE_QLORA else True,
    init_lora_weights="gaussian"
)
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    quantization_config=bnb_config if USE_QLORA else None,
)
model.add_adapter(lora_config)
model.enable_adapters()

dataset = load_dataset("csv", data_files="../dataset/train.csv")#["train"]#.select(range(100))
# split into training and validation and test
# train_test_split = dataset[].train_test_split(test_size=0.2)

# # Further split the training set to create a validation set
# train_val_split = train_test_split['train'].train_test_split(test_size=0.1)

train_dataset = dataset['train']
# val_dataset = train_val_split['test']
# test_dataset = train_test_split['test']

data_collator = MyDataCollator(processor)

training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir="./weights",
    save_strategy="steps",
    save_steps=80,
    save_total_limit=10,
    # evaluation_strategy="steps",
    # eval_steps=2,
    # evaluation_strategy="epoch",
    fp16=True,
    remove_unused_columns=False,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    # eval_dataset=val_dataset,
)

trainer.train()