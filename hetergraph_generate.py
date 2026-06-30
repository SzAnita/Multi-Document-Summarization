import json
import random
import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from hetergraph import HDSGModel, MultiNewsWrapperEval, build_vocab_and_filter, collate_hdsg_eval, decode_topk, filter_min_documents, load_glove_embeddings

def batch_preprocess_text(examples):
    raw_docs = examples['document']
    batch_clean_sentences = []
    batch_clean_articles = []
    labels = []

    for raw_text in raw_docs:
        labels.append([])
        if not raw_text:
            batch_clean_articles.append([])
            continue

        articles = raw_text.split('|||||')
        num_docs = len(articles)

        if num_docs <= 3:
            sents_per_doc_limit = 15
        else:
            sents_per_doc_limit = int(100 / num_docs)
            sents_per_doc_limit = max(sents_per_doc_limit, 10)

        clean_articles = []

        for article in articles:
            sents = nltk.sent_tokenize(article)
            clean_sents = []
            valid_count = 0

            for s in sents:
                words = s.split()
                n_words = len(words)

                if n_words > 270:
                    s = " ".join(words[:270])

                clean_sents.append(s)
                valid_count += 1

                if valid_count >= sents_per_doc_limit:
                    break
            clean_articles.append(" ".join(clean_sents))

        batch_clean_articles.append(clean_articles)

    return {'articles': batch_clean_articles, 'labels': labels}

def load_for_evaluation(vocab, glove_vectors, device, checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    model = HDSGModel(
        vocab_size=len(vocab) + 1,
        glove_weights=glove_vectors
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Found checkpoint from Epoch {checkpoint.get('epoch', '?')}")
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except RuntimeError as e:
        print("Error loading state_dict. Retrying with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()

    return model

def run_evaluation(model, test_loader, device, k=3):
    model.eval()

    predictions = []
    references = []
    sources = []

    with torch.no_grad():
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Evaluation")
        for i, batch in progress_bar:
            g, (sent_pad, sent_lens, inverse_indices), doc_indices, raw_sents_batch, raw_refs_batch = batch

            g = g.to(device)
            sent_pad, sent_lens = sent_pad.to(device), sent_lens.to(device)
            inverse_indices = inverse_indices.to(device)
            doc_indices = [d.to(device) for d in doc_indices]

            logits = model(g, (sent_pad, sent_lens, inverse_indices), doc_indices)

            cursor = 0
            for j in range(len(raw_sents_batch)):
                n_sents = len(raw_sents_batch[j])
                example_logits = logits[cursor: cursor + n_sents]
                cursor += n_sents
                generated_summary = decode_topk(example_logits, raw_sents_batch[j], top_k=10)
                clean_ref = "\n".join(nltk.sent_tokenize(raw_refs_batch[j]))
                clean_sources = ["\n".join(nltk.sent_tokenize(s)) for s in raw_sents_batch[j]]
                predictions.append(generated_summary)
                references.append(clean_ref)
                sources.append(clean_sources)
    
    return predictions, references, clean_sources

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = load_dataset("Awesome075/multi_news_parquet")
full_test = full_dataset["test"]
train_processed_full_clean = load_from_disk("preprocessed_data/train_data_full_w3")
test_processed = full_test.map(batch_preprocess_text, batched=True, batch_size=4)
train_full_clean = train_processed_full_clean.filter(filter_min_documents)
test_full_clean = test_processed.filter(filter_min_documents)
vocab = build_vocab_and_filter(train_processed_full_clean)

test_set = MultiNewsWrapperEval(test_full_clean, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)

glove_embeddings = load_glove_embeddings(vocab)
loaded_model = load_for_evaluation(vocab, glove_embeddings, device, 'checkpoint_sent_50_15_6kern_w3_full_best.pth.tar')
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
articles, generated, references = [], [], []
with open("generated_summaries.json", "r") as f:
    for line in f:
        data = json.loads(line)
        articles.append(data['article'])
        generated.append(data['generated_summary'])
        references.append(data['reference'])


with open("total_len\\test_total_len_500.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})

with open("total_len\\heter_graph_preds\\test_total_len_500_preds.json", "w") as f:
    json.dump(data, f)


with open("total_len\\test_total_len_750.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("total_len\\heter_graph_preds\\test_total_len_750_preds.json", "w") as f:
    json.dump(data, f)


with open("total_len\\test_total_len_1000.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("total_len\\heter_graph_preds\\test_total_len_1000_preds.json", "w") as f:
    json.dump(data, f)


with open("total_len\\test_total_len_1250.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("total_len\\heter_graph_preds\\test_total_len_1250_preds.json", "w") as f:
    json.dump(data, f)


with open("total_len\\test_total_len_1500.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("total_len\\heter_graph_preds\\test_total_len_1500_preds.json", "w") as f:
    json.dump(data, f)


with open("total_len\\test_total_len_rest.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("total_len\\heter_graph_preds\\test_total_len_rest_preds.json", "w") as f:
    json.dump(data, f)


with open("nr_doc\\test_2doc.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("nr_doc\\heter_graph_preds\\test_2doc_preds.json", "w") as f:
    json.dump(data, f)


with open("nr_doc\\test_3doc.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("nr_doc\\heter_graph_preds\\test_3doc_preds.json", "w") as f:
    json.dump(data, f)


with open("nr_doc\\test_4doc.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("nr_doc\\heter_graph_preds\\test_4doc_preds.json", "w") as f:
    json.dump(data, f)


with open("nr_doc\\test_5doc.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("nr_doc\\heter_graph_preds\\test_5doc_preds.json", "w") as f:
    json.dump(data, f)


with open("nr_doc\\test_6doc.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("nr_doc\\heter_graph_preds\\test_6doc_preds.json", "w") as f:
    json.dump(data, f)


with open("nr_doc\\test_rest.json", "r") as f:
    test = json.load(f)
test_set = MultiNewsWrapperEval(test, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
preds, refs, clean_sources = run_evaluation(loaded_model, test_loader, device)
data = []
for i in range(len(preds)):
    data.append({'prediction':preds[i], 'summary': refs[i]})
with open("nr_doc\\heter_graph_preds\\test_rest_preds.json", "w") as f:
    json.dump(data, f)