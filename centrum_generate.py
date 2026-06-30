import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LEDConfig,
    LEDForConditionalGeneration,
)

class CentrumDataset(Dataset):
    def __init__(self, json_filepath, tokenizer, max_source_len=1024, max_target_len=400):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        raw_docs = item['articles']
        docs = [doc.strip() for doc in raw_docs if doc.strip()]
        docs = f" {self.tokenizer.sep_token} ".join(docs)
        if not docs:
            docs = ["."]

        model_inputs = self.tokenizer(
            docs,
            max_length=self.max_source_len,
            truncation=True,
        )

        labels = self.tokenizer(
            item['summary'],
            max_length=self.max_target_len,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
def generate_and_save_predictions(model, dataloader, tokenizer, output_filepath, device='cuda'):
    model.eval()
    model.to(device)

    results = []
    gen_kwargs = {
        "max_length": 400,
        "min_length": 150,
        "num_beams": 4,
        "length_penalty": 2.0,
        "early_stopping": True,
        "no_repeat_ngram_size": 3
    }

    print(f"Starting generation for {output_filepath}...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Summaries"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            generated_token_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )

            predictions = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            labels = batch['labels'].to(device)
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            for pred, ref in zip(predictions, references):
                results.append({
                    "prediction": pred.strip(),
                    "summary": ref.strip()
                })

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(results)} predictions to {output_filepath}")

tokenizer = AutoTokenizer.from_pretrained('ratishsp/Centrum-multinews')
config=LEDConfig.from_pretrained('ratishsp/Centrum-multinews')
model = LEDForConditionalGeneration.from_pretrained('ratishsp/Centrum-multinews')

dataset = CentrumDataset("nr_doc\\test_2doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\centrum-multinews-preds\\test_2doc_preds.json")

dataset = CentrumDataset("nr_doc\\test_3doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\centrum-multinews-preds\\test_3doc_preds.json")

dataset = CentrumDataset("nr_doc\\test_4doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\centrum-multinews-preds\\test_4doc_preds.json")

dataset = CentrumDataset("nr_doc\\test_5doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\centrum-multinews-preds\\test_5doc_preds.json")

dataset = CentrumDataset("nr_doc\\test_6doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\centrum-multinews-preds\\test_6doc_preds.json")

dataset = CentrumDataset("nr_doc\\test_rest.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\centrum-multinews-preds\\test_restdoc_preds.json")

dataset = CentrumDataset("total_len\\test_total_len_500words.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\centrum-multinews-preds\\test_total_len_500words_preds.json")

dataset = CentrumDataset("total_len\\test_total_len_750words.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\centrum-multinews-preds\\test_total_len_750words_preds.json")

dataset = CentrumDataset("total_len\\test_total_len_1000words.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\centrum-multinews-preds\\test_total_len_1000words_preds.json")

dataset = CentrumDataset("total_len\\test_total_len_1250words.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\centrum-multinews-preds\\test_total_len_1250words_preds.json")

dataset = CentrumDataset("total_len\\test_total_len_1500words.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\centrum-multinews-preds\\test_total_len_1500words_preds.json")

dataset = CentrumDataset("total_len\\test_total_len_restwords.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\centrum-multinews-preds\\test_total_len_restwords_preds.json")