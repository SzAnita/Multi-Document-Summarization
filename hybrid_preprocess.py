import json
import os
from datasets import load_dataset, load_from_disk
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hetergraph import MultiNewsWrapperEval, build_vocab_and_filter, collate_hdsg_eval, decode_topk, filter_min_documents, load_glove_embeddings
from hetergraph_generate import batch_preprocess_text, load_for_evaluation


def create_dataset_for_plm(model, test_loader, device, file, k=20):
    model.eval()

    predictions = []
    references = []

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

                generated_summary = decode_topk(example_logits, raw_sents_batch[j], top_k=k)
                clean_ref = "\n".join(nltk.sent_tokenize(raw_refs_batch[j]))
                predictions.append(generated_summary)
                references.append(clean_ref) 

    data = {
        'input_text': predictions,
        'summaries': references
    }
    with open(file, "w") as f:
        f.write(json.dumps(data))

full_dataset = load_dataset("Awesome075/multi_news_parquet")
full_test = full_dataset["test"]
test_processed = full_test.map(batch_preprocess_text, batched=True, batch_size=4)
test_full_clean = test_processed.filter(filter_min_documents)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = "./preprocessed_data"
os.makedirs(save_path, exist_ok=True)
train_clean = load_from_disk(f"{save_path}/train_data_full_w3")
vocab = build_vocab_and_filter(train_clean)
glove_embeddings = load_glove_embeddings(vocab)
loaded_model = load_for_evaluation(vocab, glove_embeddings, device, 'checkpoint_sent_50_15_6kern_w3_full_best.pth.tar')
test_set = MultiNewsWrapperEval(test_full_clean, vocab)
test_loader = DataLoader(
    test_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
create_dataset_for_plm(loaded_model, test_loader, device, "multinews_data_test_hybrid.json")

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
create_dataset_for_plm(loaded_model, test_loader, device, "total_len\\hybrid\\multinews_data_test_total_len_500.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "total_len\\hybrid\\multinews_data_test_total_len_750.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "total_len\\hybrid\\multinews_data_test_total_len_1000.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "total_len\\hybrid\\multinews_data_test_total_len_1250.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "total_len\\hybrid\\multinews_data_test_total_len_1500.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "total_len\\hybrid\\multinews_data_test_total_len_rest.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "nr_doc\\hybrid\\multinews_data_test_2doc.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "nr_doc\\hybrid\\multinews_data_test_3doc.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "nr_doc\\hybrid\\multinews_data_test_4doc.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "nr_doc\\hybrid\\multinews_data_test_5doc.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "nr_doc\\hybrid\\multinews_data_test_6doc.json")


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
create_dataset_for_plm(loaded_model, test_loader, device, "nr_doc\\hybrid\\multinews_data_test_rest.json")