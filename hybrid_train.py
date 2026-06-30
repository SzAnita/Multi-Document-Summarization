from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
from tqdm import tqdm
import nltk
import os
from collections import defaultdict, Counter
import numpy as np
import string
from nltk.corpus import stopwords
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import random
import json
from datasets import Dataset
from transformers import GenerationConfig

def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=1024, truncation=True)

    labels = tokenizer(
        text_target=examples["summaries"],
        max_length=400,
        truncation=True
    )

    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nltk.download('stopwords')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

model_checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

with open("multinews_data_train_bart.json", "r") as f: 
    train_data = json.load(f)

with open("multinews_data_val_bart.json", "r") as f: 
    val_data = json.load(f)

train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

generation_config = GenerationConfig(
    max_length=400,
    min_length=150,
    no_repeat_ngram_size=3,
    num_beams=2,
)
model.generation_config = generation_config
model.config.tie_encoder_decoder = True
args = Seq2SeqTrainingArguments(
    output_dir=f"extract-then-abstract-bart-base",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=False,
    fp16=True,
    push_to_hub=False,
    generation_config=generation_config,
    load_best_model_at_end=True,     
    metric_for_best_model="eval_loss", 
    greater_is_better=False,
)
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
trainer.train()