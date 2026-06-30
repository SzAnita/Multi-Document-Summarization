from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def filter_min_documents(example):
    raw_docs = example["document"].split("|||||")
    valid_docs = [doc.strip() for doc in raw_docs if len(doc.strip()) > 0]
    text = " ".join(valid_docs)
    return len(valid_docs) >= 2 and len(text.strip()) > 0

def clean_documents(example):
    example['document'] = example['document'].replace("|||||", f" {tokenizer.sep_token} ")
    return example
class Reina():
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.retriever = None
        self.vectorizer = None
        self.tfidf_matrix = None

    def build_index(self, training_documents, training_summaries):
        self.training_keys = training_documents
        self.training_values = training_summaries
        self.vectorizer = TfidfVectorizer(max_features=50000, stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(training_documents)


    def retrieve_and_concatenate_multinews(self, input_documents_string, golden_summary, is_training=False, current_index=None):
        query_vec = self.vectorizer.transform([input_documents_string[:4000]])
        cosine_similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
        k_to_retrieve = self.top_k + 1 if is_training else self.top_k
        top_indices = cosine_similarities.argsort()[-k_to_retrieve:][::-1]
        retrieved_summaries = []
        for idx in top_indices:
            retrieved_summary = self.training_values[idx]
            
            if is_training and retrieved_summary.strip() == golden_summary.strip():
                continue 
                
            retrieved_summaries.append(retrieved_summary)
            if len(retrieved_summaries) == self.top_k:
                break
        
        source_tokens = self.tokenizer(
            input_documents_string, 
            max_length=600, 
            truncation=True, 
            add_special_tokens=False
        )
        truncated_source_text = self.tokenizer.decode(source_tokens["input_ids"])
        
        return truncated_source_text + " " + f" {tokenizer.sep_token} ".join(retrieved_summaries)

def augment_example(example, idx):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=example["document"],
        golden_summary=example['summary'],
        is_training=True, 
        current_index=idx
    )
    return {"input_text": augmented_text}

def tokenize(examples):
    model_inputs = tokenizer(
        examples["input_text"], 
        max_length=1024, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        examples["summary"], 
        max_length=400, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

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
            sents_per_doc_limit = int(50 / num_docs)
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

nltk.download("punkt_tab")

full_dataset = load_dataset("Awesome075/multi_news_parquet")
full_train = full_dataset["train"]
full_val = full_dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

filtered_train = full_train.filter(filter_min_documents)
filtered_val = full_val.filter(filter_min_documents)
filtered_test = full_test.filter(filter_min_documents)

clean_train = filtered_train.map(clean_documents)
clean_val = filtered_val.map(clean_documents)
clean_test = filtered_test.map(clean_documents)

reina = Reina()
reina.build_index(clean_train['document'], clean_train['summary'])

train_dataset = clean_train.map(augment_example, with_indices=True, batch_size=8)
val_dataset = clean_val.map(augment_example, with_indices=True, batch_size=8)
test_dataset = clean_test.map(augment_example, with_indices=True, batch_size=8)
tokenized_train = train_dataset.map(tokenize)
tokenized_val = val_dataset.map(tokenize)
tokenized_test = test_dataset.map(tokenize)
tokenized_train.save_to_disk("reina_train")
tokenized_val.save_to_disk("reina_val")
tokenized_test.save_to_disk("reina_test")

test_processed = filtered_test.map(batch_preprocess_text, batched=True, batch_size=4)

test_2doc = []
test_3doc = []
test_4doc = []
test_5doc = []
test_6doc = []
test_rest = []

for example in test_processed:
    docs = example['articles']
    if len(docs) == 2:
        test_2doc.append({'articles': example['document'], 'summary': example['summary']})
    elif len(docs) == 3:
        test_3doc.append({'articles': example['document'], 'summary': example['summary']})
    elif len(docs) == 4:
        test_4doc.append({'articles': example['document'], 'summary': example['summary']})
    elif len(docs) == 5:
        test_5doc.append({'articles': example['document'], 'summary': example['summary']})
    elif len(docs) == 6:
        test_6doc.append({'articles': example['document'], 'summary': example['summary']})
    elif len(docs) > 6:
        test_rest.append({'articles': example['document'], 'summary': example['summary']})

test_2doc_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(test_2doc)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_2doc_dict['input_text'] = augmented_texts
test_2doc_dict['summary'] = summaries

with open("nr_doc\\reina\\test_2doc.json", "w") as f:
        f.write(json.dumps(test_2doc_dict))


test_3doc_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(test_3doc)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_3doc_dict['input_text'] = augmented_texts
test_3doc_dict['summary'] = summaries

with open("nr_doc\\reina\\test_3doc.json", "w") as f:
        f.write(json.dumps(test_3doc_dict))


test_4doc_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(test_4doc)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_4doc_dict['input_text'] = augmented_texts
test_5doc_dict['summary'] = summaries

with open("nr_doc\\reina\\test_4doc.json", "w") as f:
        f.write(json.dumps(test_4doc_dict))


test_5doc_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(test_5doc)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_5doc_dict['input_text'] = augmented_texts
test_5doc_dict['summary'] = summaries

with open("nr_doc\\reina\\test_5doc.json", "w") as f:
        f.write(json.dumps(test_5doc_dict))


test_6doc_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(test_6doc)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_6doc_dict['input_text'] = augmented_texts
test_6doc_dict['summary'] = summaries

with open("nr_doc\\reina\\test_6doc.json", "w") as f:
        f.write(json.dumps(test_6doc_dict))


test_restdoc_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(test_restdoc)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_restdoc_dict['input_text'] = augmented_texts
test_restdoc_dict['summary'] = summaries

with open("nr_doc\\reina\\test_restdoc.json", "w") as f:
        f.write(json.dumps(test_restdoc_dict))


total_len_500 = []
total_len_750 = []
total_len_1000 = []
total_len_1250 = []
total_len_1500 = []
total_len_rest = []

for example in test_processed:
    total_len = 0
    article_nr = len(example['articles'])
    for article in example['articles']:
        total_len += len(nltk.word_tokenize(article))
    if total_len > 0 and total_len <= 500 and article_nr > 0:
        total_len_500.append({'articles':example['document'], 'summary':example['summary']})
    elif total_len > 500 and total_len <= 750 and article_nr > 0:
        total_len_750.append({'articles':example['document'], 'summary':example['summary']})
    elif total_len > 750 and total_len <= 1000 and article_nr > 0:
        total_len_1000.append({'articles':example['document'], 'summary':example['summary']})
    elif total_len > 1000 and total_len <= 1250 and article_nr > 0:
        total_len_1250.append({'articles':example['document'], 'summary':example['summary']})
    elif total_len > 1250 and total_len <= 1500 and article_nr > 0:
        total_len_1500.append({'articles':example['document'], 'summary':example['summary']})
    elif total_len > 1500 and article_nr > 0:
       total_len_rest.append({'articles':example['document'], 'summary':example['summary']})

test_total_len_500_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(total_len_500)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_total_len_500_dict['input_text'] = augmented_texts
test_total_len_500_dict['summary'] = summaries

with open("total_len\\test_total_len_500.json", "w") as f:
        f.write(json.dumps(test_total_len_500_dict))


test_total_len_750_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(total_len_750)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_total_len_750_dict['input_text'] = augmented_texts
test_total_len_750_dict['summary'] = summaries

with open("total_len\\test_total_len_750.json", "w") as f:
        f.write(json.dumps(test_total_len_750_dict))


test_total_len_1000_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(total_len_1000)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_total_len_1000_dict['input_text'] = augmented_texts
test_total_len_1000_dict['summary'] = summaries

with open("total_len\\test_total_len_1000.json", "w") as f:
        f.write(json.dumps(test_total_len_1000_dict))


test_total_len_1250_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(total_len_1250)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_total_len_1250_dict['input_text'] = augmented_texts
test_total_len_1250_dict['summary'] = summaries

with open("total_len\\test_total_len_1250.json", "w") as f:
        f.write(json.dumps(test_total_len_1250_dict))


test_total_len_1500_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(total_len_1500)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_total_len_1500_dict['input_text'] = augmented_texts
test_total_len_1500_dict['summary'] = summaries

with open("total_len\\test_total_len_1500.json", "w") as f:
        f.write(json.dumps(test_total_len_1500_dict))


test_total_len_rest_dict = {}
augmented_texts = []
summaries = []
for idx, row in enumerate(tqdm(total_len_rest)):
    augmented_text = reina.retrieve_and_concatenate_multinews(
        input_documents_string=row['articles'],
        is_training=False,
        current_index=idx
    )
    augmented_texts.append(augmented_text)
    summaries.append(row['summary'])
test_total_len_rest_dict['input_text'] = augmented_texts
test_total_len_rest_dict['summary'] = summaries

with open("total_len\\test_total_len_rest.json", "w") as f:
        f.write(json.dumps(test_total_len_rest_dict))