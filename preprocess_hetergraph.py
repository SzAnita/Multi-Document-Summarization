from datasets import load_dataset, concatenate_datasets, load_from_disk
from rouge_score import rouge_scorer
import multiprocessing
import torch
from tqdm import tqdm
import nltk
import os
#import dgl
from collections import defaultdict, Counter
import numpy as np
import string
from nltk.corpus import stopwords
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

def preprocess_extract_oracle_labels(example):

    raw_doc = example['document']
    articles = raw_doc.split('|||||')
    articles = [a.strip() for a in articles if a.strip()]
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2' ,'rougeL'], use_stemmer=True)

    all_sentences = []
    global_sent_idx = 0
    max_sents_per_example = 50
    max_sents_per_doc = 15
    cleaned_articles = []
    doc_sents_lst = []
    num_docs = len(articles)
    max_tokens_per_sent=140
    allowed_per_doc = max_sents_per_example
    if num_docs > 0:
        allowed_per_doc=min(max_sents_per_doc, max(max_sents_per_example//10, max_sents_per_example//num_docs))

    for article in articles:
        if global_sent_idx >= max_sents_per_example:
            break

        raw_sents = nltk.sent_tokenize(article)
        clean_sents_doc = []
        for s in raw_sents:
            words = s.split()

            if len(words) > max_tokens_per_sent:
                s = " ".join(words[:max_tokens_per_sent])

            clean_sents_doc.append(s)

        cleaned_article_text = " ".join(clean_sents_doc)
        cleaned_articles.append(cleaned_article_text)

        if len(clean_sents_doc) > allowed_per_doc:
            clean_sents_doc = clean_sents_doc[:allowed_per_doc]

        all_sentences.extend(clean_sents_doc)
        global_sent_idx += len(clean_sents_doc)
        doc_sents_lst.append(clean_sents_doc)


    summary_text = example['summary']
    selected_indices = []
    current_summary_sents = []

    search_space = list(range(len(all_sentences)))
    max_labels = 12
    while True:
        best_score = 0.0
        best_idx = -1

        if current_summary_sents:
            base_pred = "\n".join(current_summary_sents)
            base_res = scorer.score(base_pred, summary_text)
            current_fmeasure = (0.15*base_res['rouge1'].fmeasure + 0.5*base_res['rouge2'].fmeasure+0.35*base_res['rougeL'].fmeasure)
        else:
            current_fmeasure = 0.0

        for idx in search_space:
            candidate_sents = current_summary_sents + [all_sentences[idx]]
            candidate_text = "\n".join(candidate_sents)
            results = scorer.score(candidate_text, summary_text)
            score = (0.15*results['rouge1'].fmeasure + 0.5*results['rouge2'].fmeasure + 0.35*results['rougeL'].fmeasure)

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx != -1 and (best_score - current_fmeasure) > 0.001:
            selected_indices.append(best_idx)
            current_summary_sents.append(all_sentences[best_idx])
            search_space.remove(best_idx)
            if len(selected_indices) >= max_labels: break
        else:
            break

    labels = [0] * len(all_sentences)
    for idx in selected_indices:
        labels[idx] = 1

    return {'articles': cleaned_articles, 'labels': labels}


def is_valid_example(example):
    text = example['articles']

    return isinstance(text, list) and len(text) >= 2

full_dataset = load_dataset("Awesome075/multi_news_parquet")
full_train = full_dataset["train"]
full_val = full_dataset["validation"]
full_test = full_dataset["test"]

train_small1 = full_train.select(range(5000))
train_processed_small1 = train_small1.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small2 = full_train.select(range(5000, 10000))
train_processed_small2 = train_small2.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small3 = full_train.select(range(10000, 15000))
train_processed_small3 = train_small3.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small4 = full_train.select(range(15000, 20000))
train_processed_small4 = train_small4.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small5 = full_train.select(range(20000, 25000))
train_processed_small5 = train_small5.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small6 = full_train.select(range(25000, 30000))
train_processed_small6 = train_small6.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small7 = full_train.select(range(30000, 35000))
train_processed_small7 = train_small7.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small8 = full_train.select(range(35000, 40000))
train_processed_small8 = train_small8.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)
train_small9 = full_train.select(range(40000, len(full_train)))
train_processed_small9 = train_small9.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)

train_processed_full = concatenate_datasets([train_processed_small1, train_processed_small2, train_processed_small3, train_processed_small4, train_processed_small5, train_processed_small6, train_processed_small7, train_processed_small8, train_processed_small9])
val_processed_full = full_val.map(preprocess_extract_oracle_labels, batched=False, num_proc=8)

train_full_clean = train_processed_full.filter(is_valid_example)
val_full_clean = val_processed_full.filter(is_valid_example)

train_full_clean.save_to_disk("preprocessed_data/train_data_full_w3")
val_full_clean.save_to_disk("preprocessed_data/val_data_full_w3")