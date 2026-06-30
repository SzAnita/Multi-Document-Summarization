import glob
from rouge_score import rouge_scorer
import json
import pandas as pd
import numpy as np
import os

def calculate_rouge(scorer, preds, refs):
    r1, r2, rl, rlsum = [], [], [], []
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        r1.append(scores['rouge1'].fmeasure * 100)
        r2.append(scores['rouge2'].fmeasure * 100)
        rl.append(scores['rougeL'].fmeasure * 100)
        rlsum.append(scores['rougeLsum'].fmeasure * 100)
        
    return {
        'rouge1': np.mean(r1),
        'rouge2': np.mean(r2),
        'rougeL': np.mean(rl),
        'rougeLsum': np.mean(rlsum)
    }

def save_metrics(csv_path, model_name, global_scores):
    df = pd.read_csv(csv_path)

    if model_name not in df.columns:
        df[model_name] = None
    metric_mapping = {
        'R1': global_scores.get('rouge1', 0),
        'R2': global_scores.get('rouge2', 0),
        'RL': global_scores.get('rougeL', 0),
    }
    
    for metric_name, score_value in metric_mapping.items():
        df.loc[df['Metric'] == metric_name, model_name] = score_value
        
    df.to_csv(csv_path, index=False)

def evaluate_all_categories(json_folder, model, first_col_name, file_to_row_map, type):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    global_predictions = []
    global_references = []
    category_results = {}
    
    for filename, row_value in file_to_row_map.items():
        file_path = os.path.join(json_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        category_name = file_path.split('\\')[-1]
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
            
        category_scores = calculate_rouge(scorer, category_preds, category_refs)
        category_results[category_name] = {'N': N, 'scores': category_scores}
        
        print(f"Category: {category_name} (N={N})")
        print(f"  R1: {category_scores['rouge1']:.2f} | R2: {category_scores['rouge2']:.2f} | RL: {category_scores['rougeL']:.2f} |  RLsum: {category_scores['rougeLsum']:.2f}\n")
        r1_df = pd.read_csv(f"eval_results\\eval_results_{type}_r1.csv")
        r1_df.loc[r1_df[first_col_name] == row_value, model] = category_scores['rouge1']
        r1_df.to_csv(f"eval_results\\eval_results_{type}_r1.csv", index=False)
        r2_df = pd.read_csv(f"eval_results\\eval_results_nr_doc_r2.csv")
        r2_df.loc[r2_df[first_col_name] == row_value, model] = category_scores['rouge2']
        r2_df.to_csv(f"eval_results\\eval_results_{type}_r2.csv", index=False)
        rL_df = pd.read_csv(f"eval_results\\eval_results_nr_doc_rL.csv")
        rL_df.loc[r2_df[first_col_name] == row_value, model] = category_scores['rougeL']
        rL_df.to_csv(f"eval_results\\eval_results_{type}_rL.csv", index=False)


    print("Overall ROUGE score")
    total_N = len(global_predictions)
    global_scores = calculate_rouge(scorer, global_predictions, global_references)
    
    print(f"Total Samples Evaluated: {total_N}")
    print(f"Total R1: {global_scores['rouge1']:.4f}")
    print(f"Totoal R2: {global_scores['rouge2']:.4f}")
    print(f"Total RL: {global_scores['rougeL']:.4f}")

    save_metrics("eval_results\\eval_results_overall.csv", model, global_scores)
    
    return category_results, global_scores


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

nr_doc, _ = evaluate_all_categories("nr_doc\\heter_graph_preds", "Heterogenous Graph", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_all_categories("total_len\\heter_graph_preds", "Heterogenous Graph", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_all_categories("nr_doc\\hybrid_preds", "Hybrid Model", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_all_categories("total_len\\hybrid_preds", "Hybrid Model", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_all_categories("nr_doc\\pagesum_preds", "PageSum", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_all_categories("total_len\\pagesum_preds", "PageSum", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_all_categories("nr_doc\\reina_preds", "Reina", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_all_categories("total_len\\reina_preds", "Reina", "TotalWordCount", word_count_map, "total_len")
nr_doc, _ = evaluate_all_categories("nr_doc\\centrum-multinews-preds", "Centrum", "nr_doc", nr_doc_map, "nr_doc")
total_len, _ = evaluate_all_categories("total_len\\centrum-multinews-preds", "Centrum", "TotalWordCount", word_count_map, "total_len")
