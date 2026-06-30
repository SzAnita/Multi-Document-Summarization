from functools import partial
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
from rouge_score import rouge_scorer
from transformers.modeling_outputs import BaseModelOutput
from torch.optim.lr_scheduler import LambdaLR

def is_valid_example(example):
    text = example['articles']
    if isinstance(text, list):
        text = " ".join(text)

    return isinstance(text, str) and len(text.strip()) > 0

def filter_min_documents(example):
    raw_docs = example["document"].split("|||||")
    valid_docs = [doc.strip() for doc in raw_docs if len(doc.strip()) > 0]
    text = " ".join(valid_docs)
    return len(valid_docs) >= 2 and len(text.strip()) > 0

class PageSum(nn.Module):
    def __init__(self, model_name="facebook/bart-base", pad_token_id=1):
        super().__init__()
        base_bart = BartForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
        self.config = base_bart.config
        self.pad_token_id = pad_token_id
        self.bart = base_bart
        self.confidence_layer = nn.Linear(self.config.d_model, 1)
        self.bart.gradient_checkpointing_enable()
        
    def forward(self, cluster_input_ids, cluster_attention_mask, target_summary_ids):
        B, N, L = cluster_input_ids.shape
        flat_input_ids = cluster_input_ids.view(B * N, L)         
        flat_attention_mask = cluster_attention_mask.view(B * N, L) 
        decoder_input_ids = self.bart.prepare_decoder_input_ids_from_labels(target_summary_ids)
        all_hidden_states = []
        encoder = self.bart.model.encoder
        
        for i in range(0, B*N, 10):
            page_input_ids = flat_input_ids[i : i + 10]
            page_attention_mask = flat_attention_mask[i : i + 10]     
            outputs = encoder(
                input_ids=page_input_ids,
                attention_mask=page_attention_mask,
                return_dict=True
            )
            all_hidden_states.append(outputs.last_hidden_state)

        flat_hidden_states = torch.cat(all_hidden_states, dim=0)
        batched_hidden_states = flat_hidden_states.view(B, N, L, -1)
        confidence_scores = self.confidence_layer(batched_hidden_states) 
        weights = F.softmax(confidence_scores, dim=1)         
        aggregated_hidden_states = torch.sum(weights * batched_hidden_states, dim=1) 
        aggregated_mask = cluster_attention_mask.max(dim=1)[0]
        decoder_input_ids = self.bart.prepare_decoder_input_ids_from_labels(target_summary_ids)
        decoder_outputs = self.bart.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=aggregated_hidden_states,
            encoder_attention_mask=aggregated_mask,
            return_dict=True
        )
        lm_logits = self.bart.lm_head(decoder_outputs.last_hidden_state) + self.bart.final_logits_bias
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=0.1)
        loss = loss_fct(
            lm_logits.view(-1, self.config.vocab_size), 
            target_summary_ids.view(-1)
        )
        return loss

    
   
    @torch.no_grad()
    def generate_beam(self, cluster_input_ids, cluster_attention_mask, max_length=400, min_length=150, num_beams=4):
        B, N, L = cluster_input_ids.shape
        flat_input_ids = cluster_input_ids.view(B * N, L)
        flat_attention_mask = cluster_attention_mask.view(B * N, L)
        all_encoder_hidden = []
        encoder = self.bart.model.encoder
        
        for i in range(0, B * N, 10):
            page_ids = flat_input_ids[i : i + 10]
            page_mask = flat_attention_mask[i : i + 10]
            
            outputs = encoder(
                input_ids=page_ids, 
                attention_mask=page_mask, 
                return_dict=True
            )
            all_encoder_hidden.append(outputs.last_hidden_state)
            
        flat_hidden_states = torch.cat(all_encoder_hidden, dim=0)
        batched_hidden_states = flat_hidden_states.view(B, N, L, -1)
        confidence_scores = self.confidence_layer(batched_hidden_states)
        weights = F.softmax(confidence_scores, dim=1)
        aggregated_hidden_states = torch.sum(weights * batched_hidden_states, dim=1)      
        aggregated_mask = cluster_attention_mask.max(dim=1)[0] 
        encoder_outputs = BaseModelOutput(last_hidden_state=aggregated_hidden_states)
        generated_ids = self.bart.generate(
            attention_mask=aggregated_mask,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0
        )
        
        return generated_ids

class HFMultiNewsDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_docs=7, doc_max_len=1024, tgt_max_len=400, from_json=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_docs = max_docs
        self.doc_max_len = doc_max_len
        self.tgt_max_len = tgt_max_len
        self.from_json = from_json

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.from_json:
            raw_docs = item['articles']
        else:
            raw_docs = item['document'].split("|||||")
        docs = [doc.strip() for doc in raw_docs if doc.strip()]
        docs = docs[:self.max_docs]
        
        if not docs:
            docs = ["."] 

        src_enc = self.tokenizer(
            docs, 
            max_length=self.doc_max_len, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )       
        tgt_enc = self.tokenizer(
            item['summary'],
            max_length=self.tgt_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "src_input_ids": src_enc["input_ids"],
            "src_attention_mask": src_enc["attention_mask"],
            "tgt_input_ids": tgt_enc["input_ids"] 
        }

def collate_fn(batch, pad_token_id):
    max_docs = max([item["src_input_ids"].size(0) for item in batch])
    seq_len = batch[0]["src_input_ids"].size(1)
    batch_src_ids = torch.full((len(batch), max_docs, seq_len), pad_token_id, dtype=torch.long)
    batch_src_mask = torch.zeros((len(batch), max_docs, seq_len), dtype=torch.long)
    for i, item in enumerate(batch):
        num_docs = item["src_input_ids"].size(0)
        batch_src_ids[i, :num_docs, :] = item["src_input_ids"]
        batch_src_mask[i, :num_docs, :] = item["src_attention_mask"]
    
    if batch[0]["tgt_input_ids"].dim() == 2:
        batch_tgt_ids = torch.cat([item["tgt_input_ids"] for item in batch], dim=0)
    else:
        batch_tgt_ids = torch.stack([item["tgt_input_ids"] for item in batch], dim=0)
        
    return {
        "src_input_ids": batch_src_ids,
        "src_attention_mask": batch_src_mask,
        "tgt_input_ids": batch_tgt_ids
    }

def validate(model, tokenizer, val_loader, device):
    model.eval()
    total_val_loss = 0.0
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    total_rouge1, total_rouge2, total_rougeL, total_rougeLsum = 0.0, 0.0, 0.0, 0.0
    num_evaluated = 0
    print("Starting generation and evaluation...")
    preds = []
    refs = []
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")

        for step, batch in enumerate(progress_bar):
            src_ids = batch["src_input_ids"].to(device)
            src_mask = batch["src_attention_mask"].to(device)
            tgt_ids = batch["tgt_input_ids"].to(device)

            with torch.amp.autocast('cuda'):
                loss = model(
                    cluster_input_ids=src_ids,
                    cluster_attention_mask=src_mask,
                    target_summary_ids=tgt_ids
                )
                loss = loss.mean()

            total_val_loss += loss.item()

            if step%5 == 0:           
                generated_ids = model.module.generate_beam_full(
                        cluster_input_ids=src_ids,
                        cluster_attention_mask=src_mask,
                        num_beams=2
                )
                generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                gold_texts = tokenizer.batch_decode(batch["tgt_input_ids"], skip_special_tokens=True)
                clean_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in generated_texts]
                clean_refs = ["\n".join(nltk.sent_tokenize(gold.strip())) for gold in gold_texts]
                 
                for gen_text, gold_text in zip(clean_preds, clean_refs):
                        preds.append(gen_text)
                        refs.append(gold_text)
                        scores = scorer.score(target=gold_text, prediction=gen_text)
                        total_rouge1 += scores['rouge1'].fmeasure
                        total_rouge2 += scores['rouge2'].fmeasure
                        total_rougeL += scores['rougeL'].fmeasure
                        total_rougeLsum += scores['rougeLsum'].fmeasure
                        num_evaluated += 1
            if step % 10 == 0:
                progress_bar.set_postfix({"val_loss": f"{(total_val_loss / (step + 1)):.4f}"})

        r1 = (total_rouge1 / num_evaluated) * 100
        r2 = (total_rouge2 / num_evaluated) * 100
        rl = (total_rougeL / num_evaluated) * 100
        rls = (total_rougeLsum / num_evaluated) * 100
        print(f'Validation loss: {total_val_loss/len(val_loader):.4f}\n, R1: {r1:.4f}, R2: {r2:.4f}, RL: {rl:.4f}, RLsum: {rls:.4f}')
        rouge = 2*r1*r2/(r1+r2)
        print(f'Harmonic mean: {rouge:.4f}')
    return total_val_loss / len(val_loader), rouge

def get_pagesum_schedule(optimizer, warmup_steps):
    
    def lr_lambda(current_step):
        step = max(1, current_step)
        decay = step ** -0.5
        warmup = step * (warmup_steps ** -1.5)
        return min(decay, warmup)

    return LambdaLR(optimizer, lr_lambda)

def train(model, epochs=20, resume_from_ckpt=False, ckpt_path=None):
    accumulation_steps = 4

    optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    scaler = GradScaler()
    warmup_steps = 10000
    scheduler = get_pagesum_schedule(optimizer, warmup_steps)
    best_val_loss = float('inf')
    epoch_no_improve = 0
    patience = 3
    start_epoch = 0
    best_rouge = 0


    if resume_from_ckpt and os.path.exists(ckpt_path):
        print(f"Found checkpoint at {ckpt_path}. Resuming training...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        epoch_no_improve = checkpoint['epoch_no_improve']
        patience = checkpoint['patience']
        best_rouge = checkpoint['best_rouge']
        best_val_loss = checkpoint['best_val_loss']
        current_rouge = checkpoint['rouge']
        if current_rouge > best_rouge:
            best_rouge = current_rouge

        print(f"Successfully resumed. Starting from Epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(progress_bar):
            src_ids = batch["src_input_ids"].to(device)
            src_mask = batch["src_attention_mask"].to(device)
            tgt_ids = batch["tgt_input_ids"].squeeze().to(device)
            
            with torch.amp.autocast('cuda'):
                loss = model(
                    cluster_input_ids=src_ids,
                    cluster_attention_mask=src_mask,
                    target_summary_ids=tgt_ids
                )
                loss = loss.mean()
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scale_before = scaler.get_scale()
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                
            if step % 10 == 0:
                progress_bar.set_postfix({"loss": f"{(total_loss / (step + 1)):.4f}"})

        val_loss, rouge = validate(model, tokenizer, val_loader, device)
        trainable_state_dict = {
            name: param for name, param in model.module.named_parameters() if param.requires_grad
        }
        checkpoint = {
            'epoch': epoch+1,
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'rouge': rouge,
            'epoch_no_improve': epoch_no_improve,
            'patience': patience,
            'best_val_loss': best_val_loss,
            'best_rouge': best_rouge
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            epoch_no_improve = 0
            checkpoint['epoch_no_improve'] = epoch_no_improve
            checkpoint_path = "pagesum_best_val_loss-bart-base.pt"
            print(f"\nSaving model to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            
        else:
            epoch_no_improve += 1
            checkpoint['epoch_no_improve'] = epoch_no_improve
            
        if rouge > best_rouge:
            best_rouge = rouge
            checkpoint['best_rouge'] = rouge
            checkpoint_path = "pagesum_best_rouge-bart-base.pt"
            print(f"\nSaving model to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            
        

        print(f"\nSaving model to pagesum_last-bart-base-epoch.pt")
        torch.save(checkpoint, "pagesum_last-bart-base.pt")

        if epoch_no_improve == patience:
            break


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = PageSum(model_name="facebook/bart-base", pad_token_id=tokenizer.pad_token_id)
model.to(device)

full_dataset = load_dataset("Awesome075/multi_news_parquet")
full_train = full_dataset["train"]
full_val = full_dataset["validation"]
full_test = full_dataset["test"]

clean_train = full_train.filter(filter_min_documents)
clean_val = full_val.filter(filter_min_documents)
clean_test = full_test.filter(filter_min_documents)

train_dataset = HFMultiNewsDataset(clean_train, tokenizer, max_docs=10)
val_dataset = HFMultiNewsDataset(clean_val, tokenizer, max_docs=10)
custom_collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
train_loader = DataLoader(train_dataset,batch_size=4, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)

train(model, 20)

