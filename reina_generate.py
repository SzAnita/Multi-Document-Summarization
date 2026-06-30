import json
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset as HFDataset

test_dataset = load_from_disk("reina_test")

class ReinaDataset(Dataset):
    def __init__(self, json_filepath, tokenizer, max_source_len=1024, max_target_len=400):
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = HFDataset.from_dict(data)
            
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['input_text'] 
        target_text = item['summary']    
        
        model_inputs = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            truncation=True,
        )
        
        labels = self.tokenizer(
            target_text,
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

tokenizer = AutoTokenizer('facebook/bart-base')
model = AutoModelForSeq2SeqLM.from_pretrained("reina/checkpoint-8340")

dataset =ReinaDataset("nr_doc\\reina\\test_2doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\reina_preds\\test_2doc_preds.json")  

dataset =ReinaDataset("nr_doc\\reina\\test_3doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\reina_preds\\test_3doc_preds.json")  

dataset = ReinaDataset("nr_doc\\reina\\test_4doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\reina_preds\\test_4oc_preds.json")  

dataset = ReinaDataset("nr_doc\\reina\\test_5doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\reina_preds\\test_5doc_preds.json")  

dataset = ReinaDataset("nr_doc\\reina\\test_6doc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\reina_preds\\test_6doc_preds.json")  

dataset = ReinaDataset("nr_doc\\reina\\test_restdoc.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "nr_doc\\reina_preds\\test_restdoc_preds.json")  

dataset = ReinaDataset("total_len\\reina\\test_total_len_500.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\reina_preds\\test_total_len_500words_preds.json")

dataset = ReinaDataset("total_len\\reina\\test_total_len_500.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\reina_preds\\test_total_len_500words_preds.json")

dataset = ReinaDataset("total_len\\reina\\test_total_len_750.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\reina_preds\\test_total_len_750words_preds.json")

dataset =ReinaDataset("total_len\\reina\\test_total_len_1000.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\reina_preds\\test_total_len_1000words_preds.json")

dataset = ReinaDataset("nr_doc\\reina\\test_total_len_1250.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total\\reina_preds\\test_total_len_1250words_preds.json")

dataset = ReinaDataset("total_len\\reina\\test_total_len_1500.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\reina_preds\\test_total_len_1500words_preds.json")

dataset =ReinaDataset("total_len\\reina\\test_total_len_rest.json", tokenizer)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
generate_and_save_predictions(model, dataloader, tokenizer, "total_len\\reina_preds\\test_total_len_restwords_preds.json")