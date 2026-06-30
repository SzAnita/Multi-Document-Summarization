import json
from torch.utils.data import Dataset, DataLoader
import nltk
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from pagesum_train import PageSum, HFMultiNewsDataset


def generate_and_save_predictions(model, dataloader, tokenizer, output_filepath, device='cuda'):
    model.eval()
    model.to(device)
    results = []
    print(f"Starting generation for {output_filepath}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Summaries"):
            src_ids = batch["src_input_ids"].to(device)
            src_mask = batch["src_attention_mask"].to(device)
            tgt_ids = batch["tgt_input_ids"].squeeze().to(device)
            with torch.amp.autocast('cuda'):
                generated_token_ids = model.generate_beam_full(
                    cluster_input_ids=src_ids,
                    cluster_attention_mask=src_mask,
                    max_length=400,
                    min_length=150,
                    num_beams=5
                )
            
            predictions = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(tgt_ids, skip_special_tokens=True)
            clean_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
            clean_refs = ["\n".join(nltk.sent_tokenize(gold.strip())) for gold in references]
            for pred, ref in zip(clean_preds, clean_refs):
                results.append({
                    "prediction": pred,
                    "summary": ref
                })

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"Saved {len(results)} predictions to {output_filepath}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PageSum()
tokenizer = AutoTokenizer('facebook/bart-base')
checkpoint = torch.load("pagesum_best_rouge-bart-base5.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

with open("nr_doc\\test_2doc.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "nr_doc\\pagesum_preds\\test_2doc_preds.json")

with open("nr_doc\\test_3doc.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "nr_doc\\pagesum_preds\\test_3doc_preds.json")

with open("nr_doc\\test_3doc.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "nr_doc\\pagesum_preds\\test_4doc_preds.json")

with open("nr_doc\\test_5doc.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "nr_doc\\pagesum_preds\\test_5doc_preds.json")

with open("nr_doc\\test_6doc.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "nr_doc\\pagesum_preds\\test_6doc_preds.json")

with open("nr_doc\\test_rest.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "nr_doc\\pagesum_preds\\test_restdoc_preds.json")

with open("total_len\\test_total_len_500.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "total_len\\pagesum_preds\\test_500words_preds.json")

with open("total_len\\test_total_len_750.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "total_len\\pagesum_preds\\test_750words_preds.json")

with open("total_len\\test_total_len_1000.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "total_len\\pagesum_preds\\test_1000words_preds.json")

with open("total_len\\test_total_len_1250.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "total_len\\pagesum_preds\\test_1250words_preds.json")

with open("total_len\\test_total_len_1500.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "total_len\\pagesum_preds\\test_1500words_preds.json")

with open("total_len\\test_total_len_rest.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
test_dataset = HFMultiNewsDataset(data, tokenizer, max_docs=10, doc_max_len=1024, tgt_max_len=400, from_json=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
generate_and_save_predictions(model, test_loader, tokenizer, "total_len\\pagesum_preds\\test_restwords_preds.json")