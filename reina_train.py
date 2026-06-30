import random
from transformers import  DataCollatorForSeq2Seq, BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import load_dataset

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

train_dataset = load_from_disk("reina_train")
val_dataset = load_from_disk("reina_val")

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
data_collator = DataCollatorForSeq2Seq(tokenizer, model)
training_args = Seq2SeqTrainingArguments(
    output_dir="./reina",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=False, 
    fp16=True, 
    load_best_model_at_end=True,     
    metric_for_best_model="eval_loss", 
    greater_is_better=False
)
trainer = Seq2SeqTrainer(
    model=reina.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=reina.tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

train_results = trainer.train()