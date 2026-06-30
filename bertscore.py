import json
import torch
import os
import json
import pandas as pd
from bert_score import scorer as bert_scorer

device = "cuda" if torch.cuda.is_available() else "cpu"

def save_metrics(csv_path, model_name, global_scores):
    df = pd.read_csv(csv_path)

    if model_name not in df.columns:
        df[model_name] = None
    metric_mapping = {
        'BERTScore':global_scores.get('BERTScore', 0)
    }
    
    for metric_name, score_value in metric_mapping.items():
        df.loc[df['Metric'] == metric_name, model_name] = score_value
        
    df.to_csv(csv_path, index=False)

def evaluate_categories_bertscore(json_folder, model, first_col_name, file_to_row_map, type, lang="en"):
    global_predictions = []
    global_references = []
    category_results = {}
    df_path = f"eval_results\\eval_results_{type}_bertscore.csv"
    df = pd.read_csv(df_path)
   
    for filename, row_value in file_to_row_map.items():
        file_path = os.path.join(json_folder, filename)
        
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        category_name = os.path.basename(file_path)
        category_preds = []
        category_refs = []
        
        for example in data:
            pred = example['prediction'] 
            ref = example['summary']     
            category_preds.append(pred)
            category_refs.append(ref)
            global_predictions.append(pred)
            global_references.append(ref)
      
        N = len(category_preds)
        if N == 0:
            continue
        _, _, F1 = bert_scorer.score(category_preds, category_refs, lang=lang, verbose=False, device=device)
        category_bertscore_f1 = F1.mean().item()
        bertscore_f1_val = round(category_bertscore_f1 * 100, 4)
        category_results[category_name] = {'N': N, 'bert_f1': bertscore_f1_val}
        
        print(f"Category: {category_name} (N={N})")
        print(f"BERTScore F1: {bertscore_f1_val:.2f}\n")
    
        df.loc[df[first_col_name] == row_value, model] = bertscore_f1_val

    if model in df.columns:
        df[model] = pd.to_numeric(df[model], errors='coerce')
        
    df.to_csv(df_path, index=False)

    print("Overall BERTScore")
    total_N = len(global_predictions)
    
    _, _, global_F1 = bert_scorer.score(global_predictions, global_references, lang=lang, verbose=False, device=device)
    overall_f1_score = global_F1.mean().item() * 100
    
    print(f"Total Samples Evaluated: {total_N}")
    print(f"Total BERTScore F1: {overall_f1_score:.4f}")

    global_bertscore = {'BERTScore': overall_f1_score} 
    save_metrics("eval_results\\eval_results_overall.csv", model, global_bertscore)
    
    return category_results, overall_f1_score

nr_doc_map = {
    "test_2doc_preds.json": "2 documents",
    "test_3doc_preds.json": "3 documents",
    "test_4doc_preds.json": "4 documents",
    "test_5doc_preds.json": "5 documents",
    "test_6doc_preds.json": "6 documents",
    "test_restdoc_preds.json": ">6 documents"
}

word_count_map = {
    "test_total_len_500words_preds.json": "0-500 words",
    "test_total_len_750words_preds.json": "500-750 words",
    "test_total_len_1000words_preds.json": "750-1000 words",
    "test_total_len_1250words_preds.json": "1000-1250 words",
    "test_total_len_1500words_preds.json": "1250-1500 words",
    "test_total_len_restwords_preds.json": ">1500 words"
}


nr_doc, _ = evaluate_categories_bertscore("nr_doc\\heter_graph_preds", "Heterogenous Graph", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_categories_bertscore("total_len\\heter_graph_preds", "Heterogenous Graph", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_categories_bertscore("nr_doc\\hybrid_preds", "Hybrid Model", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_categories_bertscore("total_len\\hybrid_preds", "Hybrid Model", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_categories_bertscore("nr_doc\\pagesum_preds", "PageSum", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_categories_bertscore("total_len\\pagesum_preds", "PageSum", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_categories_bertscore("nr_doc\\reina_preds", "Reina", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_categories_bertscore("total_len\\reina_preds", "Reina", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_categories_bertscore("nr_doc\\centrum-multinews-preds", "Centrum", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_categories_bertscore("total_len\\centrum-multinews-preds", "Centrum", "TotalWordCount", word_count_map, "total_len")