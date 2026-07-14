# Multi-Document-Summarization
The files can be run by the following command: python [filename]
The MultiNews dataset is available from HuggingFace: https://huggingface.co/datasets/Awesome075/multi_news_parquet
The GloVe vectors can be downloaded from here: https://nlp.stanford.edu/data/glove.42B.300d.zip
To train and generate the summaries with the Heterogenous Graph the following dependencies need to be installed: pip install torch=<2.3 nltk dgl datasets tqdm
To train and generate the summaries with the Hybrid Model, PageSum, Reina ND Centrum the following dependencies need to be installed: pip install torch transformers>=5.0 nltk datasets
To run the ROUGE evaluation the rouge_score library needs to be installed: pip install rouge-score
To run the BERTScore the following dependency have to be installed: pip install bert-score
To run the SummaC evaluation the following dependencies have to be installed: pip install summac tqdm. The summac library is incompatible with transformers>=5.0