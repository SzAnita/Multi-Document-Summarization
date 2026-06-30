from summac.model_summac import SummaCConv
import json
from tqdm import tqdm
import random
import numpy as np
import pandas as pd


def evaluate_with_summac(model, source_documents: list, summaries: list):
    scores = []

    for i in tqdm(range(len(source_documents))):

        combined_source = "\n\n".join(source_documents[i])
        results = model.score([combined_source], [summaries[i]])
   
        scores.append(results["scores"][0])
        
    return scores

model_conv = SummaCConv(
    models=["vitc"], 
    bins='percentile', 
    granularity="sentence", 
    device="cuda"
)

df_summac_nr_doc = pd.read_csv("eval_results\\eval_results_nr_doc_summac.csv")
df_summac_total_len = pd.read_csv("eval_results\\eval_results_total_len_summac.csv")

with open("nr_doc\\test_2doc.json", "r", encoding="utf-8") as f:
    articles_2doc = json.load(f)
with open("nr_doc\\test_3doc.json", "r", encoding="utf-8") as f:
    articles_3doc = json.load(f)
with open("nr_doc\\test_4doc.json", "r", encoding="utf-8") as f:
    articles_4doc = json.load(f)
with open("nr_doc\\test_5doc.json", "r", encoding="utf-8") as f:
    articles_5doc = json.load(f)
with open("nr_doc\\test_6doc.json", "r", encoding="utf-8") as f:
    articles_6doc = json.load(f)
with open("nr_doc\\test_rest.json", "r", encoding="utf-8") as f:
    articles_restdoc = json.load(f)
articles_2doc_lst = []
articles_3doc_lst = []
articles_4doc_lst = []
articles_5doc_lst = []
articles_6doc_lst = []
articles_restdoc_lst = []
for item in articles_2doc:
    articles_2doc_lst.append(item['articles'])
for item in articles_3doc:
    articles_3doc_lst.append(item['articles'])
for item in articles_4doc:
    articles_4doc_lst.append(item['articles'])
for item in articles_5doc:
    articles_5doc_lst.append(item['articles'])
for item in articles_6doc:
    articles_6doc_lst.append(item['articles'])
for item in articles_restdoc:
    articles_restdoc_lst.append(item['articles'])

with open("total_len\\test_total_len_500.json", "r", encoding="utf-8") as f:
    articles_500words = json.load(f)
with open("total_len\\test_total_len_750.json", "r", encoding="utf-8") as f:
    articles_750words = json.load(f)
with open("total_len\\test_total_len_1000.json", "r", encoding="utf-8") as f:
    articles_1000words = json.load(f)
with open("total_len\\test_total_len_1250.json", "r", encoding="utf-8") as f:
    articles_1250words = json.load(f)
with open("total_len\\test_total_len_1500.json", "r", encoding="utf-8") as f:
    articles_1500words = json.load(f)
with open("total_len\\test_total_len_rest.json", "r", encoding="utf-8") as f:
    articles_restwords = json.load(f)
articles_500words_lst = []
articles_750words_lst = []
articles_1000words_lst = []
articles_1250words_lst = []
articles_1500words_lst = []
articles_restwords_lst = []
for item in articles_500words:
    articles_500words_lst.append(item['articles'])
for item in articles_750words:
    articles_750words_lst.append(item['articles'])
for item in articles_1000words:
    articles_1000words_lst.append(item['articles'])
for item in articles_1250words:
    articles_1250words_lst.append(item['articles'])
for item in articles_1500words:
    articles_1500words_lst.append(item['articles'])
for item in articles_restwords:
    articles_restwords_lst.append(item['articles'])

with open("nr_doc\\heter_graph_preds\\test_2doc_preds.json", "r", encoding="utf-8") as f:
    heter_graph_2doc = json.load(f)
heter_graph_2doc_preds = []
for item in heter_graph_2doc:
    heter_graph_2doc_preds.append(item['pred'])
heter_graph_2doc_summac =  np.mean(evaluate_with_summac(model_conv, articles_2doc_lst, heter_graph_2doc_preds))*100

with open("nr_doc\\heter_graph_preds\\test_3doc_preds.json", "r", encoding="utf-8") as f:
    heter_graph_3doc = json.load(f)
heter_graph_3doc_preds = []
for item in heter_graph_3doc:
    heter_graph_3doc_preds.append(item['pred'])
heter_graph_3doc_summac =  evaluate_with_summac(model_conv, articles_3doc_lst, heter_graph_3doc_preds)

with open("nr_doc\\heter_graph_preds\\test_4doc_preds.json", "r", encoding="utf-8") as f:
    heter_graph_4doc = json.load(f)
heter_graph_4doc_preds = []
for item in heter_graph_4doc:
    heter_graph_4doc_preds.append(item['pred'])
heter_graph_4doc_summac =  evaluate_with_summac(model_conv, articles_4doc_lst, heter_graph_4doc_preds)


with open("nr_doc\\heter_graph_preds\\test_5doc_preds.json", "r", encoding="utf-8") as f:
    heter_graph_5doc = json.load(f)
heter_graph_5doc_preds = []
for item in heter_graph_5doc:
    heter_graph_5doc_preds.append(item['pred'])
heter_graph_5doc_summac =  evaluate_with_summac(model_conv, articles_5doc_lst, heter_graph_5doc_preds)

with open("nr_doc\\heter_graph_preds\\test_6doc_preds.json", "r", encoding="utf-8") as f:
    heter_graph_6doc = json.load(f)
heter_graph_6doc_preds = []
for item in heter_graph_6doc:
    heter_graph_6doc_preds.append(item['pred'])
heter_graph_6doc_summac =  evaluate_with_summac(model_conv, articles_6doc_lst, heter_graph_6doc_preds)

with open("nr_doc\\heter_graph_preds\\test_rest_preds.json", "r", encoding="utf-8") as f:
    heter_graph_restdoc = json.load(f)
heter_graph_restdoc_preds = []
for item in heter_graph_restdoc:
    heter_graph_restdoc_preds.append(item['pred'])
heter_graph_restdoc_summac =  evaluate_with_summac(model_conv, articles_restdoc_lst, heter_graph_6doc_preds)

df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "2 documents", "Heterogenous Graph"] = np.mean(heter_graph_2doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "3 documents", "Heterogenous Graph"] = np.mean(heter_graph_3doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "4 documents", "Heterogenous Graph"] = np.mean(heter_graph_4doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "5 documents", "Heterogenous Graph"] = np.mean(heter_graph_5doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "6 documents", "Heterogenous Graph"] = np.mean(heter_graph_6doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == ">6 documents", "Heterogenous Graph"] = np.mean(heter_graph_restdoc_summac)*100

with open("total_len\\heter_graph_preds\\test_total_len_500words_preds.json", "r", encoding="utf-8") as f:
    heter_graph_500words = json.load(f)
heter_graph_500words_preds = []
for item in heter_graph_500words:
    heter_graph_500words_preds.append(item['pred'])
heter_graph_500words_summac =  evaluate_with_summac(model_conv, articles_500words_lst, heter_graph_500words_preds)

with open("total_len\\heter_graph_preds\\test_total_len_750words_preds.json", "r", encoding="utf-8") as f:
    heter_graph_750words = json.load(f)
heter_graph_750words_preds = []
for item in heter_graph_750words:
    heter_graph_750words_preds.append(item['pred'])
heter_graph_750words_summac =  evaluate_with_summac(model_conv, articles_750words_lst, heter_graph_750words_preds)

with open("total_len\\heter_graph_preds\\test_total_len_1000words_preds.json", "r", encoding="utf-8") as f:
    heter_graph_1000words = json.load(f)
heter_graph_1000words_preds = []
for item in heter_graph_1000words:
    heter_graph_1000words_preds.append(item['pred'])
heter_graph_1000words_summac =  evaluate_with_summac(model_conv, articles_1000words_lst, heter_graph_1000words_preds)

with open("total_len\\heter_graph_preds\\test_total_len_1250words_preds.json", "r", encoding="utf-8") as f:
    heter_graph_1250words = json.load(f)
heter_graph_1250words_preds = []
for item in heter_graph_1250words:
    heter_graph_1250words_preds.append(item['pred'])
heter_graph_1250words_summac =  evaluate_with_summac(model_conv, articles_1250words_lst, heter_graph_1250words_preds)

with open("total_len\\heter_graph_preds\\test_total_len_1500words_preds.json", "r", encoding="utf-8") as f:
    heter_graph_1500words = json.load(f)
heter_graph_1500words_preds = []
for item in heter_graph_1500words:
    heter_graph_1500words_preds.append(item['pred'])
heter_graph_1500words_summac =  evaluate_with_summac(model_conv, articles_1500words_lst, heter_graph_1500words_preds)

with open("total_len\\heter_graph_preds\\test_total_len_restwords_preds.json", "r", encoding="utf-8") as f:
    heter_graph_restwords = json.load(f)
heter_graph_restwords_preds = []
for item in heter_graph_restwords:
    heter_graph_restwords_preds.append(item['pred'])
heter_graph_restwords_summac =  evaluate_with_summac(model_conv, articles_restwords_lst, heter_graph_restwords_preds)

df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "0-500 words", "Heterogenous Graph"] = np.mean(heter_graph_500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "500-750 words", "Heterogenous Graph"] = np.mean(heter_graph_750words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "750-1000 words", "Heterogenous Graph"] = np.mean(heter_graph_1000words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1000-1250 words", "Heterogenous Graph"] = np.mean(heter_graph_1250words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1250-1500 words", "Heterogenous Graph"] = np.mean(heter_graph_1500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == ">1500 words", "Heterogenous Graph"] = np.mean(heter_graph_restwords_summac)*100

with open("nr_doc\\hybrid_preds\\test_2doc_preds.json", "r", encoding="utf-8") as f:
    hybrid_2doc = json.load(f)
hybrid_2doc_preds = []
for item in hybrid_2doc:
    hybrid_2doc_preds.append(item['prediction'])
hybrid_2doc_summac =  evaluate_with_summac(model_conv, articles_2doc_lst, hybrid_2doc_preds)

with open("nr_doc\\hybrid_preds\\test_2doc_preds.json", "r", encoding="utf-8") as f:
    hybrid_3doc = json.load(f)
hybrid_3doc_preds = []
for item in hybrid_3doc:
    hybrid_3doc_preds.append(item['prediction'])
hybrid_3doc_summac =  evaluate_with_summac(model_conv, articles_3doc_lst, hybrid_3doc_preds)

with open("nr_doc\\hybrid_preds\\test_4doc_preds.json", "r", encoding="utf-8") as f:
    hybrid_4doc = json.load(f)
hybrid_4doc_preds = []
for item in hybrid_4doc:
    hybrid_4doc_preds.append(item['prediction'])
hybrid_4doc_summac =  evaluate_with_summac(model_conv, articles_4doc_lst, hybrid_4doc_preds)

with open("nr_doc\\hybrid_preds\\test_5doc_preds.json", "r", encoding="utf-8") as f:
    hybrid_5doc = json.load(f)
hybrid_5doc_preds = []
for item in hybrid_5doc:
    hybrid_5doc_preds.append(item['prediction'])
hybrid_5doc_summac =  evaluate_with_summac(model_conv, articles_5doc_lst, hybrid_5doc_preds)

with open("nr_doc\\bart_preds\\test_6doc.json", "r", encoding="utf-8") as f:
    hybrid_6doc = json.load(f)
hybrid_6doc_preds = []
for item in hybrid_6doc:
    hybrid_6doc_preds.append(item['prediction'])
hybrid_6doc_summac =  evaluate_with_summac(model_conv, articles_6doc_lst, hybrid_6doc_preds)

with open("nr_doc\\bart_preds\\test_rest.json", "r", encoding="utf-8") as f:
    hybrid_restdoc = json.load(f)
hybrid_restdoc_preds = []
for item in hybrid_restdoc:
    hybrid_restdoc_preds.append(item['prediction'])
hybrid_restdoc_summac =  evaluate_with_summac(model_conv, articles_restdoc_lst, hybrid_restdoc_preds)

df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "2 documents", "Hybrid Model"] = np.mean(hybrid_2doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "3 documents", "Hybrid Model"] = np.mean(hybrid_3doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "4 documents", "Hybrid Model"] = np.mean(hybrid_4doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "5 documents", "Hybrid Model"] = np.mean(hybrid_5doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "6 documents", "Hybrid Model"] = np.mean(hybrid_6doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == ">6 documents", "Hybrid Model"] = np.mean(hybrid_restdoc_summac)*100

with open("total_len\\hybrid_preds\\test_total_len_500words_preds.json", "r", encoding="utf-8") as f:
    hybrid_500words = json.load(f)
hybrid_500words_preds = []
for item in hybrid_500words:
    hybrid_500words_preds.append(item['prediction'])
hybrid_500words_summac =  evaluate_with_summac(model_conv, articles_500words_lst, hybrid_500words_preds)

with open("total_len\\hybrid_preds\\test_total_len_750words_preds.json", "r", encoding="utf-8") as f:
    hybrid_750words = json.load(f)
hybrid_750words_preds = []
for item in hybrid_750words:
    hybrid_750words_preds.append(item['prediction'])
hybrid_750words_summac =  evaluate_with_summac(model_conv, articles_750words_lst, hybrid_750words_preds)

with open("total_len\\hybrid_preds\\test_total_len_1000words_preds.json", "r", encoding="utf-8") as f:
    hybrid_1000words = json.load(f)
hybrid_1000words_preds = []
for item in hybrid_1000words:
    hybrid_1000words_preds.append(item['prediction'])
hybrid_1000words_summac =  evaluate_with_summac(model_conv, articles_1000words_lst, hybrid_1000words_preds)

with open("total_len\\hybrid_preds\\test_total_len_1250words_preds.json", "r", encoding="utf-8") as f:
    hybrid_1250words = json.load(f)
hybrid_1250words_preds = []
for item in hybrid_1250words:
    hybrid_1250words_preds.append(item['prediction'])
hybrid_1250words_summac =  evaluate_with_summac(model_conv, articles_1250words_lst, hybrid_1250words_preds)

with open("total_len\\hybrid_preds\\test_total_len_1500words_preds.json", "r", encoding="utf-8") as f:
    hybrid_1500words = json.load(f)
hybrid_1500words_preds = []
for item in hybrid_1500words:
    hybrid_1500words_preds.append(item['prediction'])
hybrid_1500words_summac =  evaluate_with_summac(model_conv, articles_1500words_lst, hybrid_1500words_preds)

with open("total_len\\hybrid_preds\\test_total_len_restwords_preds.json", "r", encoding="utf-8") as f:
    hybrid_restwords = json.load(f)
hybrid_restwords_preds = []
for item in hybrid_restwords:
    hybrid_restwords_preds.append(item['prediction'])
hybrid_restwords_summac =  evaluate_with_summac(model_conv, articles_restwords_lst, hybrid_restwords_preds)

df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "0-500 words", "Hybrid Model"] = np.mean(hybrid_500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "500-750 words", "Hybrid Model"] = np.mean(hybrid_750words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "750-1000 words", "Hybrid Model"] = np.mean(hybrid_1000words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1000-1250 words", "Hybrid Model"] = np.mean(hybrid_1250words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1250-1500 words", "Hybrid Model"] = np.mean(hybrid_1500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == ">1500 words", "Hybrid Model"] = np.mean(hybrid_restwords_summac)*100

with open("nr_doc\\pagesum_preds\\test_2doc_preds.json", "r", encoding="utf-8") as f:
    pagesum_2doc = json.load(f)
pagesum_2doc_preds = []
for item in pagesum_2doc:
    pagesum_2doc_preds.append(item['prediction'])
pagesum_2doc_summac =  evaluate_with_summac(model_conv, articles_2doc_lst, pagesum_2doc_preds)

with open("nr_doc\\pagesum_preds\\test_3doc_preds.json", "r", encoding="utf-8") as f:
    pagesum_3doc = json.load(f)
pagesum_3doc_preds = []
for item in pagesum_3doc:
    pagesum_3doc_preds.append(item['prediction'])
pagesum_3doc_summac =  evaluate_with_summac(model_conv, articles_3doc_lst, pagesum_3doc_preds)

with open("nr_doc\\pagesum_preds\\test_4doc_preds.json", "r", encoding="utf-8") as f:
    pagesum_4doc = json.load(f)
pagesum_4doc_preds = []
for item in pagesum_4doc:
    pagesum_4doc_preds.append(item['prediction'])
pagesum_4doc_summac =  evaluate_with_summac(model_conv, articles_4doc_lst, pagesum_4doc_preds)

with open("nr_doc\\pagesum_preds\\test_5doc_preds.json", "r", encoding="utf-8") as f:
    pagesum_5doc = json.load(f)
pagesum_5doc_preds = []
for item in pagesum_5doc:
    pagesum_5doc_preds.append(item['prediction'])
pagesum_5doc_summac =  evaluate_with_summac(model_conv, articles_5doc_lst, pagesum_5doc_preds)

with open("nr_doc\\pagesum_preds\\test_6doc_preds.json", "r", encoding="utf-8") as f:
    pagesum_6doc = json.load(f)
pagesum_6doc_preds = []
for item in pagesum_6doc:
    pagesum_6doc_preds.append(item['prediction'])
pagesum_6doc_summac =  evaluate_with_summac(model_conv, articles_6doc_lst, pagesum_6doc_preds)

with open("nr_doc\\pagesum_preds\\test_restdoc_preds.json", "r", encoding="utf-8") as f:
    pagesum_restdoc = json.load(f)
pagesum_restdoc_preds = []
for item in pagesum_restdoc:
    pagesum_restdoc_preds.append(item['prediction'])
pagesum_restdoc_summac =  evaluate_with_summac(model_conv, articles_restdoc_lst, pagesum_restdoc_preds)

df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "2 documents", "PageSum"] = np.mean(pagesum_2doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "3 documents", "PageSum"] = np.mean(pagesum_3doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "4 documents", "PageSum"] = np.mean(pagesum_4doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "5 documents", "PageSum"] = np.mean(pagesum_5doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "6 documents", "PageSum"] = np.mean(pagesum_6doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == ">6 documents", "PageSum"] = np.mean(pagesum_restdoc_summac)*100

with open("total_len\\pagesum_preds\\test_total_len_500words_preds.json", "r", encoding="utf-8") as f:
    pagesum_500words = json.load(f)
pagesum_500words_preds = []
for item in pagesum_500words:
    pagesum_500words_preds.append(item['prediction'])
pagesum_500words_summac =  evaluate_with_summac(model_conv, articles_500words_lst, pagesum_500words_preds)

with open("total_len\\pagesum_preds\\test_total_len_750words_preds.json", "r", encoding="utf-8") as f:
    pagesum_750words = json.load(f)
pagesum_750words_preds = []
for item in pagesum_750words:
    pagesum_750words_preds.append(item['prediction'])
pagesum_750words_summac =  evaluate_with_summac(model_conv, articles_750words_lst, pagesum_750words_preds)

with open("total_len\\pagesum_preds\\test_total_len_1000words_preds.json", "r", encoding="utf-8") as f:
    pagesum_1000words = json.load(f)
pagesum_1000words_preds = []
for item in pagesum_1000words:
    pagesum_1000words_preds.append(item['prediction'])
pagesum_1000words_summac =  evaluate_with_summac(model_conv, articles_1000words_lst, pagesum_1000words_preds)

with open("total_len\\pagesum_preds\\test_total_len_1250words_preds.json", "r", encoding="utf-8") as f:
    pagesum_1250words = json.load(f)
pagesum_1250words_preds = []
for item in pagesum_1250words:
    pagesum_1250words_preds.append(item['prediction'])
pagesum_1250words_summac =  evaluate_with_summac(model_conv, articles_1250words_lst, pagesum_1250words_preds)

with open("total_len\\pagesum_preds\\test_total_len_1500words_preds.json", "r", encoding="utf-8") as f:
    pagesum_1500words = json.load(f)
pagesum_1500words_preds = []
for item in pagesum_1500words:
    pagesum_1500words_preds.append(item['prediction'])
pagesum_1500words_summac =  evaluate_with_summac(model_conv, articles_1500words_lst, pagesum_1500words_preds)

with open("total_len\\pagesum_preds\\test_total_len_restwords_preds.json", "r", encoding="utf-8") as f:
    pagesum_restwords = json.load(f)
pagesum_restwords_preds = []
for item in pagesum_restwords:
    pagesum_restwords_preds.append(item['prediction'])
pagesum_restwords_summac =  evaluate_with_summac(model_conv, articles_restwords_lst, pagesum_restwords_preds)

df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "0-500 words", "PageSum"] = np.mean(pagesum_500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "500-750 words", "PageSum"] = np.mean(pagesum_750words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "750-1000 words", "PageSum"] = np.mean(pagesum_1000words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1000-1250 words", "PageSum"] = np.mean(pagesum_1250words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1250-1500 words", "PageSum"] = np.mean(pagesum_1500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == ">1500 words", "PageSum"] = np.mean(pagesum_restwords_summac)*100

with open("nr_doc\\reina_preds\\test_2doc_preds.json", "r", encoding="utf-8") as f:
    reina_2doc = json.load(f)
reina_2doc_preds = []
for item in reina_2doc:
    reina_2doc_preds.append(item['prediction'])
reina_2doc_summac = evaluate_with_summac(model_conv, articles_2doc_lst, reina_2doc_preds)

with open("nr_doc\\reina_preds\\test_3doc_preds.json", "r", encoding="utf-8") as f:
    reina_3doc = json.load(f)
reina_3doc_preds = []
for item in reina_3doc:
    reina_3doc_preds.append(item['prediction'])
reina_3doc_summac = evaluate_with_summac(model_conv, articles_3doc_lst, reina_3doc_preds)

with open("nr_doc\\reina_preds\\test_4doc_preds.json", "r", encoding="utf-8") as f:
    reina_4doc = json.load(f)
reina_4doc_preds = []
for item in reina_4doc:
    reina_4doc_preds.append(item['prediction'])
reina_4doc_summac = evaluate_with_summac(model_conv, articles_4doc_lst, reina_4doc_preds)

with open("nr_doc\\reina_preds\\test_5doc_preds.json", "r", encoding="utf-8") as f:
    reina_5doc = json.load(f)
reina_5doc_preds = []
for item in reina_5doc:
    reina_5doc_preds.append(item['prediction'])
reina_5doc_summac = evaluate_with_summac(model_conv, articles_5doc_lst, reina_5doc_preds)

with open("nr_doc\\reina_preds\\test_6doc_preds.json", "r", encoding="utf-8") as f:
    reina_6doc = json.load(f)
reina_6doc_preds = []
for item in reina_6doc:
    reina_6doc_preds.append(item['prediction'])
reina_6doc_summac = evaluate_with_summac(model_conv, articles_6doc_lst, reina_6doc_preds)

with open("nr_doc\\reina_preds\\test_restdoc_preds.json", "r", encoding="utf-8") as f:
    reina_restdoc = json.load(f)
reina_restdoc_preds = []
for item in reina_restdoc:
    reina_restdoc_preds.append(item['prediction'])
reina_restdoc_summac = evaluate_with_summac(model_conv, articles_restdoc_lst, reina_restdoc_preds)

df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "2 documents", "Reina"] = np.mean(reina_2doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "3 documents", "Reina"] = np.mean(reina_3doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "4 documents", "Reina"] = np.mean(reina_4doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "5 documents", "Reina"] = np.mean(reina_5doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "6 documents", "Reina"] = np.mean(reina_5doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == ">6 documents", "Reina"] = np.mean(reina_restdoc_summac)*100

with open("total_len\\reina_preds\\test_total_len_500words_preds.json", "r", encoding="utf-8") as f:
    reina_500words = json.load(f)
reina_500words_preds = []
for item in reina_500words:
    reina_500words_preds.append(item['prediction'])
reina_500words_summac = evaluate_with_summac(model_conv, articles_500words_lst, reina_500words_preds)

with open("total_len\\reina_preds\\test_total_len_750words_preds.json", "r", encoding="utf-8") as f:
    reina_750words = json.load(f)
reina_750words_preds = []
for item in reina_750words:
    reina_750words_preds.append(item['prediction'])
reina_750words_summac = evaluate_with_summac(model_conv, articles_750words_lst, reina_750words_preds)

with open("total_len\\reina_preds\\test_total_len_1000words_preds.json", "r", encoding="utf-8") as f:
    reina_1000words = json.load(f)
reina_1000words_preds = []
for item in reina_1000words:
    reina_1000words_preds.append(item['prediction'])
reina_1000words_summac = evaluate_with_summac(model_conv, articles_1000words_lst, reina_1000words_preds)

with open("total_len\\reina_preds\\test_total_len_1250words_preds.json", "r", encoding="utf-8") as f:
    reina_1250words = json.load(f)
reina_1250words_preds = []
for item in reina_1250words:
    reina_1250words_preds.append(item['prediction'])
reina_1250words_summac = evaluate_with_summac(model_conv, articles_1250words_lst, reina_1250words_preds)

with open("total_len\\reina_preds\\test_total_len_1500words_preds.json", "r", encoding="utf-8") as f:
    reina_1500words = json.load(f)
reina_1500words_preds = []
for item in reina_1500words:
    reina_1500words_preds.append(item['prediction'])
reina_1500words_summac = evaluate_with_summac(model_conv, articles_1500words_lst, reina_1500words_preds)

with open("total_len\\reina_preds\\test_total_len_restwords_preds.json", "r", encoding="utf-8") as f:
    reina_restwords = json.load(f)
reina_restwords_preds = []
for item in reina_restwords:
    reina_restwords_preds.append(item['prediction'])
reina_restwords_summac = evaluate_with_summac(model_conv, articles_restwords_lst, reina_restwords_preds)

df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "0-500 words", "Reina"] = np.mean(reina_500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "500-750 words", "Reina"] = np.mean(reina_750words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "750-1000 words", "Reina"] = np.mean(reina_1000words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1000-1250 words", "Reina"] = np.mean(reina_1250words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1250-1500 words", "Reina"] = np.mean(reina_1500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == ">1500 words", "Reina"] = np.mean(reina_restwords_summac)*100

with open("nr_doc\\centrum-multinews-preds\\test_2doc-preds.json", "r", encoding="utf-8") as f:
    centrum_2doc = json.load(f)
centrum_2doc_preds = []
for item in centrum_2doc:
    centrum_2doc_preds.append(item['prediction'])
centrum_2doc_summac = evaluate_with_summac(model_conv, articles_2doc_lst, centrum_2doc_preds)

with open("nr_doc\\centrum-multinews-preds\\test_3doc-preds.json", "r", encoding="utf-8") as f:
    centrum_3doc = json.load(f)
centrum_3doc_preds = []
for item in centrum_3doc:
    centrum_3doc_preds.append(item['prediction'])
centrum_3doc_summac = evaluate_with_summac(model_conv, articles_3doc_lst, centrum_3doc_preds)

with open("nr_doc\\centrum-multinews-preds\\test_4doc-preds.json", "r", encoding="utf-8") as f:
    centrum_4doc = json.load(f)
centrum_4doc_preds = []
for item in centrum_4doc:
    centrum_4doc_preds.append(item['prediction'])
centrum_4doc_summac = evaluate_with_summac(model_conv, articles_4doc_lst, centrum_4doc_preds)

with open("nr_doc\\centrum-multinews-preds\\test_5doc_preds.json", "r", encoding="utf-8") as f:
    centrum_5doc = json.load(f)
centrum_5doc_preds = []
for item in centrum_5doc:
    centrum_5doc_preds.append(item['prediction'])
centrum_5doc_summac = evaluate_with_summac(model_conv, articles_5doc_lst, centrum_5doc_preds)

with open("nr_doc\\centrum-multinews-preds\\test_6doc_preds.json", "r", encoding="utf-8") as f:
    centrum_6doc = json.load(f)
centrum_6doc_preds = []
for item in centrum_6doc:
    centrum_6doc_preds.append(item['prediction'])
centrum_6doc_summac = evaluate_with_summac(model_conv, articles_6doc_lst, centrum_6doc_preds)

with open("nr_doc\\centrum-multinews-preds\\test_restdoc_preds.json", "r", encoding="utf-8") as f:
    centrum_restdoc = json.load(f)
centrum_restdoc_preds = []
for item in centrum_restdoc:
    centrum_restdoc_preds.append(item['prediction'])
centrum_restdoc_summac = evaluate_with_summac(model_conv, articles_restdoc_lst, centrum_restdoc_preds)

df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "2 documents", "Centrum"] = np.mean(centrum_2doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "3 documents", "Centrum"] = np.mean(centrum_3doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "4 documents", "Centrum"] = np.mean(centrum_4doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "5 documents", "Centrum"] = np.mean(centrum_5doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == "6 documents", "Centrum"] = np.mean(centrum_6doc_summac)*100
df_summac_nr_doc.loc[df_summac_nr_doc["nr_doc"] == ">6 documents", "Centrum"] = np.mean(centrum_restdoc_summac)*100

with open("total_len\\centrum-multinews-preds\\test_total_len_500words_preds.json", "r", encoding="utf-8") as f:
    centrum_500words = json.load(f)
centrum_500words_preds = []
for item in centrum_500words:
    centrum_500words_preds.append(item['prediction'])
centrum_500words_summac = evaluate_with_summac(model_conv, articles_500words_lst, centrum_500words_preds)

with open("total_len\\centrum-multinews-preds\\test_total_len_750words_preds.json", "r", encoding="utf-8") as f:
    centrum_750words = json.load(f)
centrum_750words_preds = []
for item in centrum_750words:
    centrum_750words_preds.append(item['prediction'])
centrum_750words_summac = evaluate_with_summac(model_conv, articles_750words_lst, centrum_750words_preds)

with open("total_len\\centrum-multinews-preds\\test_total_len_1000words_preds.json", "r", encoding="utf-8") as f:
    centrum_1000words = json.load(f)
centrum_1000words_preds = []
for item in centrum_1000words:
    centrum_1000words_preds.append(item['prediction'])
centrum_1000words_summac = evaluate_with_summac(model_conv, articles_1000words_lst, centrum_1000words_preds)

with open("total_len\\centrum-multinews-preds\\test_total_len_1250words_preds.json", "r", encoding="utf-8") as f:
    centrum_1250words = json.load(f)
centrum_1250words_preds = []
for item in centrum_1250words:
    centrum_1250words_preds.append(item['prediction'])
centrum_1250words_summac = evaluate_with_summac(model_conv, articles_1250words_lst, centrum_1250words_preds)

with open("total_len\\centrum-multinews-preds\\test_total_len_1500words_preds.json", "r", encoding="utf-8") as f:
    centrum_1500words = json.load(f)
centrum_1500words_preds = []
for item in centrum_1500words:
    centrum_1500words_preds.append(item['prediction'])
centrum_1500words_summac = evaluate_with_summac(model_conv, articles_1500words_lst, centrum_1500words_preds)

with open("total_len\\centrum-multinews-preds\\test_total_len_restwords_preds.json", "r", encoding="utf-8") as f:
    centrum_restwords = json.load(f)
centrum_restwords_preds = []
for item in centrum_restwords:
    centrum_restwords_preds.append(item['prediction'])
centrum_restwords_summac = evaluate_with_summac(model_conv, articles_restwords_lst, centrum_restwords_preds)

df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "0-500 words", "Centrum"] = np.mean(centrum_500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "500-750 words", "Centrum"] = np.mean(centrum_750words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "750-1000 words", "Centrum"] = np.mean(centrum_1000words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1000-1250 words", "Centrum"] = np.mean(centrum_1250words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == "1250-1500 words", "Centrum"] = np.mean(centrum_1500words_summac)*100
df_summac_total_len.loc[df_summac_total_len["TotalWordCount"] == ">1500 words", "Centrum"] = np.mean(centrum_restwords_summac)*100

df_summac_nr_doc.to_csv("eval_results\\eval_results_nr_doc_summac.csv", index=False)
df_summac_total_len.to_csv("eval_results\\eval_results_total_len_summac.csv", index=False)