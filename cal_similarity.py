import os
import sys
import logging
import json
import re
import tqdm
import copy
import pandas as pd
import torch
import numpy as np
import argparse

from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--icl_mode", type=str, default="random_each"
)
parser.add_argument(
    "--task", type=str, default="ATSC"
)
parser.add_argument(
    "--dataset", type=str, default="rest"
)
parser.add_argument(
    "--device", type=str, default="cuda:0"
)
parser.add_argument(
    "--seed", type=int, default=123
)
parser.add_argument(
    "--train_idx", type=int, default=1, help="Index of training data"
)
args = parser.parse_args()

def cal_bm25(train_examples, test_examples):
    corpus = [txt for txt in train_examples]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    # 对于test_examples的每一个查询，得到相关的文档
    bm25_json = {}
    for i, example in enumerate(test_examples):
        query = example.split(" ")
        doc_scores = bm25.get_scores(query)
        indices = list(np.argsort(doc_scores)[::-1])
        bm25_json[f"{i}"] = [f"{idx}" for idx in indices]
    return bm25_json

def calculate_similarity(tensor_A, tensor_B):
    # 计算两个tensor之间的余弦相似度
    norm_A = torch.nn.functional.normalize(tensor_A, p=2, dim=1)
    norm_B = torch.nn.functional.normalize(tensor_B, p=2, dim=1)
    similarity_matrix = torch.mm(norm_A, norm_B.t())
    return similarity_matrix

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def evaluate(model, tokenizer, examples, out_dir):
    out_reps = []
    with torch.no_grad():
        for sentence in tqdm.tqdm(examples):
            encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(args.device)       
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Perform pooling
            rep = mean_pooling(model_output, encoded_input['attention_mask'])
            out_reps.append(rep)
    all_tensor = torch.cat(out_reps, dim=0)
    print(all_tensor.shape)
    # torch.save(all_tensor, out_dir)
    return all_tensor
    
def evaluate_mmr(train_examples, test_sentence, lambd=0.5):
    from bert_score import score
    # score
    P, R, F1 = score([test_sentence] * len(train_examples), train_examples, model_type="bert-base-uncased", verbose=False, device=args.device, batch_size=512*4, nthreads=16*4)
    # Map each training example to its F1 score
    train_scores = {idx: f1_score for idx, f1_score in enumerate(F1.tolist())}
    T = []
    while len(T) < len(train_examples):
        mmr_score = float('-inf')
        selected_idx = None
        for idx, f1_score in train_scores.items():
            if idx not in T:
                relevance = lambd * f1_score
                diversity = max((1 - lambd) * train_scores[j] for j in T) if T else 0
                mmr_score_tmp = relevance - diversity
                # find the maximum mmr score
                if mmr_score_tmp > mmr_score:
                    mmr_score = mmr_score_tmp
                    selected_idx = idx

        if selected_idx is not None:
            T.append(selected_idx)
        else:
            break
    return T
        
# main
if __name__ == "__main__":
    # load data
    train = pd.read_csv(f"./dataset/{args.task}/{args.dataset}/train_{args.train_idx}.csv")
    test = pd.read_csv(f"./dataset/{args.task}/{args.dataset}/test.csv")
    # random_each
    if args.icl_mode == "random_each":
        print("Random each")
        task = args.task
        dataset = args.dataset
        train_idx = args.train_idx
        example_ids = {}
        my_list = train.index.tolist()
        for i in range(len(test)):
            example_ids[i] = random.sample(my_list, 300)
        if not os.path.exists(f"./ICL_examples/{task}/{dataset}"):
            os.makedirs(f"./ICL_examples/{task}/{dataset}")
        writeJson(f"./ICL_examples/{task}/{dataset}/random_each_{train_idx}.json", example_ids, encoding="utf-8")
        
    # bm25
    elif args.icl_mode == "bm25":
        print("BM25")
        task = args.task
        dataset = args.dataset
        train_idx = args.train_idx
        train_examples = []
        for k in range(len(train)):
            tokens = train.iloc[k]["sentence"]
            train_examples.append(tokens)
        test_examples = []
        for k in range(len(test)):
            tokens = test.iloc[k]["sentence"]
            test_examples.append(tokens)
        result_json = cal_bm25(train_examples, test_examples)
        if not os.path.exists(f"./ICL_examples/{task}/{dataset}"):
            os.makedirs(f"./ICL_examples/{task}/{dataset}")
        writeJson(f"./ICL_examples/{task}/{dataset}/bm25_{train_idx}.json", result_json, encoding="utf-8")
        
    # sbert
    elif args.icl_mode == "sbert":
        print("SBERT")
        task = args.task
        dataset = args.dataset
        train_idx = args.train_idx
        hf_model = "sentence-transformers/paraphrase-mpnet-base-v2"
        # load model
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModel.from_pretrained(hf_model)
        model.to(args.device)
        # embedding
        train_examples = train['sentence'].tolist()
        test_examples = test['sentence'].tolist()
        if not os.path.exists(f"./sbert_embedding/{task}/{dataset}/"):
            os.makedirs(f"./sbert_embedding/{task}/{dataset}/")
        sbert_trains = evaluate(model, tokenizer, train_examples, out_dir=f"./sbert_embedding/{task}/{dataset}/trains_{train_idx}.pt")
        sbert_evals = evaluate(model, tokenizer, test_examples, out_dir=f"./sbert_embedding/{task}/{dataset}/test.pt")
        # sim
        similarity_matrix = calculate_similarity(sbert_evals, sbert_trains)
        result_json = {}
        for i, similarity_scores in enumerate(similarity_matrix):
            sorted_indices = torch.argsort(similarity_scores, descending=True)
            sorted_indices_list = sorted_indices.tolist()
            result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]

        if not os.path.exists(f"./ICL_examples/{task}/{dataset}"):
                os.makedirs(f"./ICL_examples/{task}/{dataset}")

        writeJson(f"./ICL_examples/{task}/{dataset}/sbert_{train_idx}.json", result_json, encoding="utf-8")
        
    # kmeans
    elif args.icl_mode == "kmeans":
        print("Kmeans")
        task = args.task
        dataset = args.dataset
        train_idx = args.train_idx
        hf_model = "sentence-transformers/all-MiniLM-L6-v2" # sentence-transformers/all-MiniLM-L6-v2
        # load model
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModel.from_pretrained(hf_model)
        model.to(args.device)
        # embedding
        train_examples = train['sentence'].tolist()
        test_examples = test['sentence'].tolist()
        if not os.path.exists(f"./kmeans_embedding/{task}/{dataset}/"):
            os.makedirs(f"./kmeans_embedding/{task}/{dataset}/")
        kmeans_trains = evaluate(model, tokenizer, train_examples, out_dir=f"./kmeans_embedding/{task}/{dataset}/trains_{train_idx}.pt").cpu().numpy()
        # Perform KMeans clustering with a loop to ensure min_samples_per_cluster
        n_clusters = 4
        min_samples_per_cluster = 30
        for rs in range(25):  # Limit the number of attempts to avoid an infinite loop
            kmeans = KMeans(n_clusters=n_clusters, random_state=rs).fit(kmeans_trains)
            labels = kmeans.labels_
            cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
            # Check if all clusters have the minimum required samples
            if all(size >= min_samples_per_cluster for size in cluster_sizes):
                break
        # Group train indices by their assigned cluster
        cluster_indices = {i: np.where(kmeans.labels_ == i)[0] for i in range(n_clusters)}
        result_json = {}
        # assert
        for cluster in range(n_clusters):
            assert len(cluster_indices[cluster]) >= min_samples_per_cluster
        # distribute test examples to each cluster
        for i, _, in enumerate(test_examples):
            selected_samples = {}
            for cluster in range(n_clusters):
                temp_indices = copy.deepcopy(cluster_indices[cluster])
                random.shuffle(temp_indices)
                # Select the first sample from the shuffled list for this cluster
                selected_samples[f"cluster_{cluster}"] = [str(idx) for idx in temp_indices]
            result_json[f"{i}"] = []
            for ii in range(min_samples_per_cluster):
                for cluster in range(n_clusters):
                    result_json[f"{i}"].append(selected_samples[f"cluster_{cluster}"][ii])
        if not os.path.exists(f"./ICL_examples/{task}/{dataset}"):
            os.makedirs(f"./ICL_examples/{task}/{dataset}")
        writeJson(f"./ICL_examples/{task}/{dataset}/kmeans_{train_idx}.json", result_json, encoding="utf-8")
    
    # mmr
    elif args.icl_mode == "mmr":
        print("MMR")
        task = args.task
        dataset = args.dataset
        train_idx = args.train_idx
        train_examples = train['sentence'].tolist()
        test_examples = test['sentence'].tolist()
        if not os.path.exists(f"./mmr_embedding/{task}/{dataset}/"):
            os.makedirs(f"./mmr_embedding/{task}/{dataset}/")
        result_json = {}
        test_examples = tqdm.tqdm(test_examples)
        for i, test_exp in enumerate(test_examples):
            sorted_indices_list = evaluate_mmr(train_examples, test_exp)
            result_json[f"{i}"] = [f"{idx}" for idx in sorted_indices_list]
        if not os.path.exists(f"./ICL_examples/{task}/{dataset}"):
                os.makedirs(f"./ICL_examples/{task}/{dataset}")
        # print(result_json)
        writeJson(f"./ICL_examples/{task}/{dataset}/mmr_{train_idx}.json", result_json, encoding="utf-8")
    