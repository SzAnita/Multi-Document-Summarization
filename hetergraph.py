from datasets import load_dataset, load_from_disk
from rouge_score import rouge_scorer
import random
import torch
from tqdm import tqdm
import nltk
import os
import dgl
from collections import defaultdict, Counter
import numpy as np
import string
from nltk.corpus import stopwords
from torch import optim, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from torch.nn.utils.rnn import pad_sequence


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

def build_vocab_and_filter(dataset, vocab_size=50000, prune_ratio=0.1):
    print("Building Vocabulary and Computing Global TF-IDF...")

    stop_words = set(stopwords.words('english'))
    punct = set(string.punctuation)
    doc_freq = Counter()
    total_docs = 0
    term_freq = Counter()

    for example in dataset:
        full_text = " ".join(example['articles']).lower()
        tokens = full_text.split()
        term_freq.update(tokens)
        unique_tokens = set(tokens)
        doc_freq.update(unique_tokens)
        total_docs += 1

    tfidf_scores = {}
    for w, tf in term_freq.items():
        if w in stop_words or any(p in w for p in punct):
            continue

        df = doc_freq[w]
        idf = np.log(total_docs / (df + 1))
        tfidf_scores[w] = tf * idf

    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    cutoff_index = int(len(sorted_words) * (1.0 - prune_ratio))
    pruned_list = sorted_words[:cutoff_index]
    final_vocab_list = pruned_list[:vocab_size]
    vocab = {w: i + 2 for i, (w, score) in enumerate(final_vocab_list)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    print(f"Final Vocab Size: {len(vocab)} (Original: {len(tfidf_scores)})")
    return vocab

def load_glove_embeddings(vocab, glove_path="glove.42B.300d.txt", embed_dim=300):
    print(f"Loading GloVe vectors from {glove_path}...")

    matrix_len = len(vocab) + 1
    weights_matrix = np.zeros((matrix_len, embed_dim))

    found_count = 0
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]

            if word in vocab:
                try:
                    vector = np.asarray(values[1:], dtype='float32')
                    idx = vocab[word]
                    weights_matrix[idx] = vector
                    found_count += 1
                except ValueError:
                    # Skip broken lines (rare)
                    continue

    print(f"Loaded {found_count} words from GloVe out of {len(vocab)} in vocab.")

    return torch.tensor(weights_matrix, dtype=torch.float32)

def build_hetero_graph_multidoc(articles, vocab, max_sents=50, max_sents_doc=15):

    edges_doc_word = []
    edges_sent_word = []
    doc_word_tfs = []
    sent_word_tfs = []
    word_degree_counts = defaultdict(int)
    unique_words_in_graph = set()
    global_sent_idx = 0
    num_docs = len(articles)
    allowed_sents_per_doc = max_sents_doc
    allowed_sents_per_doc = min(max_sents_doc, max_sents // max(num_docs,1))
    doc_indices = []
    valid_sent_tokens = []
    max_doc_len = 1000
    raw_sentences = []
    for doc_idx, doc_text in enumerate(articles):
        if global_sent_idx >= max_sents:
            break

        doc_text = doc_text.strip()
        if not doc_text: continue

        raw_sentences = nltk.sent_tokenize(doc_text)

        if len(raw_sentences) > allowed_sents_per_doc:
            raw_sentences = raw_sentences[:allowed_sents_per_doc]

        doc_text_truncated = " ".join(raw_sentences)
        current_doc_sent_indices = []
        doc_tokens = doc_text_truncated.replace('.', ' ').lower().split()

        if len(doc_tokens) > max_doc_len:
            doc_tokens = doc_tokens[:max_doc_len]
            doc_text = " ".join(doc_tokens)

        doc_len = len(doc_tokens)
        doc_counts = Counter(doc_tokens)
        unique_words_in_doc = set()

        for w, count in doc_counts.items():
            if w in vocab:
                wid = vocab[w]

                edges_doc_word.append((doc_idx, wid))
                tf = count / max(doc_len, 1)
                doc_word_tfs.append(tf)
                unique_words_in_doc.add(wid)
                unique_words_in_graph.add(wid)

        for wid in unique_words_in_doc:
            word_degree_counts[wid] += 1

        for sent_text in raw_sentences:
            if not sent_text.strip():
                continue

            sent_tokens = sent_text.lower().split()
            valid_sent_tokens.append(sent_tokens)
            current_doc_sent_indices.append(global_sent_idx)
            sent_len = len(sent_tokens)
            sent_counts = Counter(sent_tokens)
            unique_words_in_sent = set()

            for w, count in sent_counts.items():
                if w in vocab:
                    wid = vocab[w]
                    edges_sent_word.append((global_sent_idx, wid))
                    tf = count / max(sent_len, 1)
                    sent_word_tfs.append(tf)
                    unique_words_in_sent.add(wid)
                    unique_words_in_graph.add(wid)

            for wid in unique_words_in_sent:
                word_degree_counts[wid] += 1

            global_sent_idx += 1

        doc_indices.append(torch.tensor(current_doc_sent_indices, dtype=torch.long))

    if global_sent_idx == 0:
        global_sent_idx = 1

    final_dw_weights = []
    for i, (doc_idx, wid) in enumerate(edges_doc_word):
        tf = doc_word_tfs[i]
        degree = word_degree_counts[wid]
        idf = 1.0 / (degree + 1e-6)
        final_dw_weights.append(tf * idf)

    final_sw_weights = []
    for i, (sent_idx, wid) in enumerate(edges_sent_word):
        tf = sent_word_tfs[i]
        degree = word_degree_counts[wid]
        idf = 1.0 / (degree + 1e-6)
        final_sw_weights.append(tf * idf)

    active_global_ids = sorted(list(unique_words_in_graph))
    global_to_local = {gid: i for i, gid in enumerate(active_global_ids)}
    num_active_words = len(active_global_ids)

    if edges_doc_word:
        src_d = [e[0] for e in edges_doc_word]
        dst_w_d = [global_to_local[e[1]] for e in edges_doc_word]
    else:
        src_d, dst_w_d = [], []

    if edges_sent_word:
        src_s = [e[0] for e in edges_sent_word]
        dst_w_s = [global_to_local[e[1]] for e in edges_sent_word]
    else:
        src_s, dst_w_s = [], []

    graph_data = {
        ('doc', 'has', 'word'): (torch.tensor(src_d, dtype=torch.long),
                                 torch.tensor(dst_w_d, dtype=torch.long)),
        ('word', 'in_doc', 'doc'): (torch.tensor(dst_w_d, dtype=torch.long),
                                torch.tensor(src_d, dtype=torch.long)),
        ('sentence', 'contains', 'word'): (torch.tensor(src_s, dtype=torch.long),
                                           torch.tensor(dst_w_s, dtype=torch.long)),
        ('word', 'in_sent', 'sentence'): (torch.tensor(dst_w_s, dtype=torch.long),
                                     torch.tensor(src_s, dtype=torch.long))
    }
    g = dgl.heterograph(graph_data, num_nodes_dict={
        'doc': max(len(articles), 1),
        'sentence': global_sent_idx,
        'word': num_active_words
    })

    dw_tensor = torch.tensor(final_dw_weights, dtype=torch.float32)
    g.edges['has'].data['tfidf'] = dw_tensor
    g.edges[('word', 'in_doc', 'doc')].data['tfidf'] = dw_tensor

    sw_tensor = torch.tensor(final_sw_weights, dtype=torch.float32)
    g.edges['contains'].data['tfidf'] = sw_tensor
    g.edges[('word', 'in_sent', 'sentence')].data['tfidf'] = sw_tensor
    g.nodes['word'].data['id'] = torch.tensor(active_global_ids, dtype=torch.long)

    return g, doc_indices, valid_sent_tokens

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = F.relu(self.w_1(x))
        output = self.dropout(self.w_2(output))
        return self.layer_norm(output + residual)

class HeteroGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, edge_dim=50, ffn_inner=512, feat_drop=0.1, attn_drop=0.1, alpha=0.2, residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.residual = residual
        self.total_dim = out_dim * num_heads

        self.W = nn.Linear(in_dim, self.total_dim, bias=False)
        self.edge_projector = nn.Linear(edge_dim, num_heads)

        if self.total_dim != in_dim:
            self.fc = nn.Linear(in_dim, self.total_dim)
        else:
            self.fc = nn.Identity()

        self.attn_l = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.ffn = PositionwiseFeedForward(self.total_dim, ffn_inner)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'W'):
            nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if hasattr(self, 'edge_projector'):
            nn.init.xavier_normal_(self.edge_projector.weight, gain=gain)
        if hasattr(self, 'fc') and isinstance(self.fc, nn.Linear):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)

    def edge_attention_message(self, edges):
        # GAT Score: (a_l * h_i) + (a_r * h_j)
        a = edges.src['el'] + edges.dst['er']

        # Add Projected Edge Weight (TF-IDF)
        # tfidf is (E, 1), projected to (E, Heads)
        if 'edge_feat' in edges.data:
            # Project 50-dim vector to [Heads] scores
            e_score = self.edge_projector(edges.data['edge_feat'])
            # Add to attention: (Batch, Heads, 1) + (Batch, Heads) -> broadcast
            a = a + e_score.unsqueeze(-1)
        return {'a': self.leaky_relu(a), 'ft': edges.src['ft']}

    def forward_one_step(self, g, edge_type, src_type, dst_type, inputs):
        h_src = inputs[src_type]
        h_dst = inputs[dst_type]

        # Projection
        feat_src = self.fc(h_src).view(-1, self.num_heads, self.out_dim)
        feat_dst = self.fc(h_dst).view(-1, self.num_heads, self.out_dim)

        # Prepare Attention Terms
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        g.nodes[src_type].data['ft'] = feat_src
        g.nodes[src_type].data['el'] = el
        g.nodes[dst_type].data['er'] = er

        # Compute unnormalized attention scores ('a') on edges
        if g.num_edges(edge_type) > 0:
            g.apply_edges(self.edge_attention_message, etype=edge_type)

            # Normalize scores using edge_softmax
            e_logits = g.edges[edge_type].data['a']
            e_logits = torch.clamp(e_logits, min=-10.0, max=10.0)

            g.edges[edge_type].data['a_norm'] = edge_softmax(g[edge_type], e_logits)
            # Aggregate weighted features
            # u_mul_e: Multiply src feature ('ft') by edge weight ('a_norm')
            g.update_all(
                fn.u_mul_e('ft', 'a_norm', 'm'),
                fn.sum('m', 'h'),
                etype=edge_type
            )
        else:
            zeros = torch.zeros(
                (g.num_nodes(dst_type), self.num_heads, self.out_dim),
                device=g.device
            )
            g.nodes[dst_type].data['h'] = zeros

        rst = g.nodes[dst_type].data.pop('h').view(-1, self.num_heads * self.out_dim)

        if h_dst.shape[1] == rst.shape[1]:
            rst = rst + h_dst

        if hasattr(self, 'res_fc'):
            rst = rst + self.res_fc(inputs[dst_type])

        rst_unsqueezed = rst.unsqueeze(1)
        rst_out = self.ffn(rst_unsqueezed)
        rst = rst_out.squeeze(1)
        rst = self.ffn(rst)

        return rst

class HDSGModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=64, lstm_hidden_dim=128, edge_dim=50, num_heads=8, ffn_inner_dim=512, glove_weights=None):
        super().__init__()

        self.word_proj = nn.Linear(embed_dim, hidden_dim)
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        if glove_weights is not None:
            self.word_embed = nn.Embedding.from_pretrained(glove_weights, freeze=False)

        self.cnn_kernel_sizes = [2, 3, 4, 5, 6, 7]
        num_filters = hidden_dim // len(self.cnn_kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in self.cnn_kernel_sizes
        ])
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden_dim, bidirectional=True, batch_first=True)

        total_cnn_dim = num_filters * len(self.cnn_kernel_sizes)
        total_enc_dim = total_cnn_dim + lstm_hidden_dim * 2
        self.sent_adapter = nn.Linear(total_enc_dim, hidden_dim)

        self.edge_encoder = nn.Linear(1, edge_dim)
        self.gat_layer = HeteroGATLayer(hidden_dim, (hidden_dim//num_heads), num_heads, edge_dim, ffn_inner_dim)
        self.classifier = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(p=0.2)


    def encode_sentences(self, sent_pad, sent_lens, inverse_indices):
        max_valid_id = self.word_embed.num_embeddings - 1
        sent_pad = torch.clamp(sent_pad, max=max_valid_id)
        x = self.word_embed(sent_pad)
        x_perm = x.permute(0, 2, 1).contiguous()
        max_kernel = max(self.cnn_kernel_sizes)

        if x_perm.size(2) < max_kernel:
            diff = max_kernel - x_perm.size(2)
            x_perm = F.pad(x_perm, (0, diff))
            x_perm = x_perm.contiguous()

        conv_outs = [F.relu(conv(x_perm)) for conv in self.convs]
        conv_outs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_outs]

        packed = nn.utils.rnn.pack_padded_sequence(x, sent_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bilstm(packed)
        lstm_feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        cnn_feat = torch.cat(conv_outs, 1)

        final_sent_feat = torch.cat([cnn_feat, lstm_feat], dim=1)
        final_sent_feat = self.sent_adapter(final_sent_feat)
        final_sent_feat = final_sent_feat[inverse_indices]

        return final_sent_feat

    def forward(self, g, sentence_input, doc_indices):

        word_ids = g.nodes['word'].data['id']
        max_valid_id = self.word_embed.num_embeddings - 1
        word_ids = torch.clamp(word_ids, max=max_valid_id)
        h_word = self.word_proj(self.word_embed(word_ids))
        h_sent = self.encode_sentences(*sentence_input)
        h_docs = []
        if len(doc_indices) > 0:
            for indices in doc_indices:
                if len(indices) == 0:
                    h_docs.append(torch.zeros_like(h_sent[0]))
                else:
                    h_docs.append(torch.mean(h_sent[indices], dim=0))
            h_doc = torch.stack(h_docs)
        else:
            h_doc = torch.zeros((1, h_sent.shape[1]), device=h_sent.device)

        num_graph_docs = g.num_nodes('doc')
        num_feat_docs = h_doc.shape[0]

        if num_feat_docs < num_graph_docs:
            diff = num_graph_docs - num_feat_docs
            pad = torch.zeros((diff, h_doc.shape[1]), device=h_doc.device)
            h_doc = torch.cat([h_doc, pad], dim=0)
        elif num_feat_docs > num_graph_docs:
            h_doc = h_doc[:num_graph_docs]

        inputs = {'word': h_word, 'sentence': h_sent, 'doc': h_doc}

        for etype in g.etypes:
            if 'tfidf' in g.edges[etype].data:
                tfidf = g.edges[etype].data['tfidf'].unsqueeze(-1)
                g.edges[etype].data['edge_feat'] = F.relu(self.edge_encoder(tfidf))

        new_doc = self.gat_layer.forward_one_step(g, 'in_doc', 'word', 'doc', inputs)
        new_sent = self.gat_layer.forward_one_step(g, 'in_sent', 'word', 'sentence', inputs)
        inputs['doc'] = new_doc
        inputs['sentence'] = new_sent
        h_w_from_d = self.gat_layer.forward_one_step(g, 'has', 'doc', 'word', inputs)
        h_w_from_s = self.gat_layer.forward_one_step(g, 'contains', 'sentence', 'word', inputs)
        inputs['word'] = h_w_from_d + h_w_from_s
        final_sent = inputs['sentence']
        final_doc = inputs['doc']

        doc_features_expanded = []
        if len(doc_indices) > 0:
            for i, indices in enumerate(doc_indices):
                if i < len(final_doc):
                    n_sents = len(indices)
                    if n_sents > 0:
                        doc_vec = final_doc[i].view(1, -1).expand(n_sents, -1)
                        doc_features_expanded.append(doc_vec)

        if doc_features_expanded:
            doc_features_expanded = torch.cat(doc_features_expanded, dim=0)
        else:
            doc_features_expanded = torch.zeros_like(final_sent)

        cat_features = torch.cat([final_sent, doc_features_expanded], dim=1)

        return torch.clamp(self.classifier(cat_features).view(-1), min=-100.0, max=100.0)

def decode_topk(logits, raw_sentences, top_k=9):
    scores = torch.sigmoid(logits).view(-1).cpu().numpy()
    if len(scores) <= top_k:
        selected_indices = list(range(len(scores)))
    else:
        selected_indices = np.argsort(scores)[-top_k:]
        selected_indices = selected_indices.tolist()

    selected_indices.sort()
    final_summary = "\n".join([raw_sentences[i] for i in selected_indices])

    return final_summary

def validate_with_rouge(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    with torch.no_grad():
        progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Validation")
        for i, batch in progress_bar:
            g, (sent_pad, sent_lens, inverse_indices), doc_indices, labels, raw_sents_batch, raw_refs_batch = batch

            g = g.to(device)
            sent_pad, sent_lens = sent_pad.to(device), sent_lens.to(device)
            inverse_indices = inverse_indices.to(device)
            doc_indices = [d.to(device) for d in doc_indices]
            labels = labels.to(device)
            logits = model(g, (sent_pad, sent_lens, inverse_indices), doc_indices)
            min_len = min(len(logits), len(labels))
            loss = criterion(logits[:min_len], labels[:min_len])
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().view(-1)
            cursor = 0
            for j in range(len(raw_sents_batch)):
                num_sents = len(raw_sents_batch[j])
                example_logits = logits[cursor: cursor + num_sents]
                cursor += num_sents
                pred_summary = decode_topk(example_logits, raw_sents_batch[j], top_k=10)
                ref_summary = "\n".join(nltk.sent_tokenize(raw_refs_batch[j]))
                scores = scorer.score(ref_summary, pred_summary)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_loss = total_loss / len(loader)
    avg_r1 = np.mean(rouge1_scores) if rouge1_scores else 0.0
    avg_r2 = np.mean(rouge2_scores) if rouge2_scores else 0.0
    avg_rL = np.mean(rougeL_scores) if rougeL_scores else 0.0
    weighted_score = (avg_r1*0.1 + avg_r2*0.6 + avg_rL*0.3)

    print(f"Val Loss: {avg_loss:.4f} | Weighted Score: {weighted_score:.4f}")
    print(f"(R1: {avg_r1:.4f} | R2: {avg_r2:.4f} | RL: {avg_rL:.4f})")
    model.train()
    return weighted_score, avg_loss

def save_checkpoint(state, filename="checkpoint_sent_50_15_6kern_w3_full_best.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)

def train(model, train_loader, val_loader, epochs=5, resume_from_checkpoint=False, resume_path=None):

    best_model = model
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1,verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    accumulation_steps = 4
    patience = 3
    best_rouge_score = 0
    epochs_no_improve = 0
    start_epoch = 0

    if resume_from_checkpoint:
        if resume_path is not None and os.path.exists(resume_path):
            print(f"Resuming from {resume_path}...")
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            epochs_no_improve = checkpoint['epochs_no_improvement']
            best_rouge_score = checkpoint['best_rouge_score']
            print(f"Resuming at Epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in progress_bar:
            g, labels, (sent_pad, sent_lens, inverse_indices), doc_indices = batch
            g = g.to(device)
            sent_pad = sent_pad.to(device)
            sent_lens = sent_lens.to(device)
            labels = labels.to(device)
            doc_indices = [d.to(device) for d in doc_indices]
            inverse_indices = inverse_indices.to(device)
            logits = model(g, (sent_pad, sent_lens, inverse_indices), doc_indices)
            min_len = min(len(logits), len(labels))
            loss = criterion(logits[:min_len], labels[:min_len])
            loss = loss/accumulation_steps
            loss.backward()
            real_batch_loss = loss.item() * accumulation_steps
            epoch_loss += real_batch_loss


            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"Batch Loss": f"{real_batch_loss:.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Average Train Loss: {avg_train_loss:.6f}")

        rouge_score, val_loss = validate_with_rouge(model, val_loader, criterion, device)
        print('Val loss: ', val_loss)

        checkpoint_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'epochs_no_improvement': epochs_no_improve,
                'best_rouge_score': best_rouge_score
            }

        if rouge_score > best_rouge_score:
            best_rouge_score = rouge_score
            epochs_no_improve = 0
            
            save_checkpoint(checkpoint_state)
            best_model = model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
        checkpoint_state['epochs_no_improvement'] = epochs_no_improve
        save_checkpoint(checkpoint_state, "checkpoint_last.pth.tar")
        scheduler.step(rouge_score)
    return best_model

class MultiNewsWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab):
        self.dataset = hf_dataset
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        articles = item['articles']
        labels = torch.tensor(item['labels'], dtype=torch.float)
        g, doc_indices, valid_sent_tokens = build_hetero_graph_multidoc(articles, self.vocab)
        sent_tensors = []
        for tokens in valid_sent_tokens:
            token_ids = [self.vocab.get(w, 1) for w in tokens] # 1 is UNK
            if not token_ids: token_ids = [0]
            sent_tensors.append(torch.tensor(token_ids, dtype=torch.long))
        if not sent_tensors:
            sent_tensors.append(torch.tensor([0], dtype=torch.long))

        return g, doc_indices, sent_tensors, labels

def collate_hdsg(batch):
    max_sent_len = 140
    graphs, doc_indices, sentences, labels = zip(*batch)
    batched_g = dgl.batch(graphs)
    batched_labels = torch.cat(labels)
    all_sentences_flat = [s for doc in sentences for s in doc]
    sent_pad = pad_sequence(all_sentences_flat, batch_first=True, padding_value=0)

    if sent_pad.size(1) > max_sent_len:
        sent_pad = sent_pad[:, :max_sent_len]

    sent_lens = (sent_pad != 0).sum(dim=1)
    sent_lens_sorted, perm_idx = sent_lens.sort(0, descending=True)
    sent_pad_sorted = sent_pad[perm_idx]
    _, inverse_indices = perm_idx.sort(0)

    new_doc_indices = []
    offset = 0
    for doc_sents, doc_inds in zip(sentences, doc_indices):
        batch_doc_inds = [d + offset for d in doc_inds]
        new_doc_indices.extend(batch_doc_inds)
        offset += len(doc_sents)

    return batched_g, batched_labels, (sent_pad_sorted, sent_lens_sorted, inverse_indices), new_doc_indices

class MultiNewsWrapperEval(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, vocab):
        self.dataset = hf_dataset
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        articles = item['articles']
        reference = item['summary']

        g, doc_indices, valid_sent_tokens = build_hetero_graph_multidoc(articles, self.vocab)

        sent_tensors = []
        raw_sents = []

        for tokens in valid_sent_tokens:
            raw_sents.append(" ".join(tokens))

            token_ids = [self.vocab.get(w, 1) for w in tokens]
            if not token_ids: token_ids = [0]
            sent_tensors.append(torch.tensor(token_ids, dtype=torch.long))

        if not sent_tensors:
            sent_tensors.append(torch.tensor([0], dtype=torch.long))
            raw_sents.append("")

        return g, doc_indices, sent_tensors, raw_sents, reference

def collate_hdsg_eval(batch):
    max_sent_len=270
    graphs, doc_indices_list, sent_tensors_list, raw_sents_list, refs_list = zip(*batch)
    batched_graph = dgl.batch(graphs)
    all_sents_flat = [s for doc in sent_tensors_list for s in doc]
    sent_pad = pad_sequence(all_sents_flat, batch_first=True, padding_value=0)

    if sent_pad.size(1) > max_sent_len:
        sent_pad = sent_pad[:, :max_sent_len]

    sent_lens = (sent_pad != 0).sum(dim=1)
    sent_lens_sorted, perm_idx = sent_lens.sort(0, descending=True)
    sent_pad_sorted = sent_pad[perm_idx]
    _, inverse_indices = perm_idx.sort(0)

    new_doc_indices = []
    offset = 0
    for doc_sents, doc_inds in zip(sent_tensors_list, doc_indices_list):
        batch_doc_inds = [d + offset for d in doc_inds]
        new_doc_indices.extend(batch_doc_inds)
        offset += len(doc_sents)


    return batched_graph, (sent_pad_sorted, sent_lens_sorted, inverse_indices), new_doc_indices, raw_sents_list, refs_list


torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nltk.download('stopwords')
nltk.download('punkt_tab')
full_dataset = load_dataset("Awesome075/multi_news_parquet")
full_train = full_dataset["train"]
full_val = full_dataset["validation"]
full_test = full_dataset["test"]
train_processed_full_clean = load_from_disk("preprocessed_data/train_data_full_w3")
val_processed_full_clean = load_from_disk("preprocessed_data/val_data_full_w3")
train_clean = train_processed_full_clean.filter(filter_min_documents)
val_clean = val_processed_full_clean.filter(filter_min_documents)
vocab_train_full = build_vocab_and_filter(train_processed_full_clean)
glove_embeddings = load_glove_embeddings(vocab_train_full)
model = HDSGModel(50000, glove_weights=glove_embeddings).to(device)


val_set = MultiNewsWrapperEval(val_clean, vocab_train_full)
val_loader = DataLoader(
    val_set,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_hdsg_eval,
    num_workers=0
)
train_set = MultiNewsWrapper(train_clean, vocab_train_full)
train_loader = DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_hdsg,
    num_workers=0
)

model = train(model, train_loader, val_loader, epochs=20)