import os
import torch
import datasets
import transformers
import random
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import argparse

from utils.utils import *
from utils.data_utils import *
from utils.generate import LLMgenerate

from torch.utils.data import DataLoader
from ICL.templates import *

from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", 
    type=str, 
    default="llama-2-13b-chat",  
    help="LLM directory / name")
parser.add_argument(
    "--task",
    type=str,
    choices=["ATSC", "SC", "EMO"],
    help="Define task name",
)
parser.add_argument(
    "--dataset",
    type=str,
    help="Dataset to test on",
)
parser.add_argument(
    "--train_examples", type=int, default=300, help="Number of training data"
)
parser.add_argument(
    "--train_idx", type=int, default=1, help="Index of training data"
)
parser.add_argument(
    "--seed", type=int, default=123, help="Random seed"
)
parser.add_argument(
    "--run_pre", action="store_true", default=False, help="Run prior prediction generation"
)
parser.add_argument(
    "--run_icl", action="store_true", default=False, help="Run ICL baseline"
)
parser.add_argument(
    "--icl_mode", type=str, default="random_each"
)
parser.add_argument(
    "--batch_size", type=int, default=8, help="Batch size for Inference"
)
parser.add_argument(
    "--load_bit", type=str, default="fp16"
)
parser.add_argument(
    "--device", type=str, default="cuda:0"
)
parser.add_argument(
    "--mode", type=str, default="train", help="running mode"
)
parser.add_argument(
    "--max_len", type=int, default=600
)
parser.add_argument(
    "--write_to_train", action="store_true", default=False
)
args = parser.parse_args()

    
def format_eval_output(rows):
    idx, tokens, labels, outputs = zip(*rows)
    idx = np.vstack(idx)
    tokens = np.vstack(tokens)
    labels = np.vstack(labels)
    outputs = np.vstack(outputs)
    results_df = pd.DataFrame()
    results_df["id"] = idx.reshape(-1).tolist()
    results_df["input_all_tokens"] = tokens.reshape(-1).tolist()
    results_df["label"] = labels.reshape(-1).tolist()
    results_df["outputs"] = outputs.reshape(-1).tolist()
    return results_df

if __name__ == "__main__":
    # seed & log
    seed_everything(seed=args.seed)
    # basic info
    print(f"LLM: {args.model}")
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Running-Mode: {args.mode}")
    print(f"Loading Tokenizer")
    print(f"Number of Training examples: {args.train_examples}")
    tokenizer = Tokenizer4LLM(args.max_len, args.model, llm=True)
    dataset_name = args.dataset
    task_name = args.task
    # load dataset
    train_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/train_{args.train_idx}.csv")
    test_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/test.csv")
    
    if args.run_pre:
        if args.mode == "train":
            test_df = train_df
        prompt = icl_instruction_prompt[dataset_name]
        TestDataset = ClassificationDataset(
                        test_df, args, prompt,
                        get_input_template, 
                        tokenizer=tokenizer
                    )
        
        test_loader = DataLoader(
                        TestDataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False
                    )  
        
        print("Loading LLM")
        all_labels = label_space[dataset_name]
        LLM = LLMgenerate(args, tokenizer, all_labels=all_labels)
        # Results
        rows = []
        print("Start Generation with the LLM")
        
        if not os.path.exists(f"./result/{args.model}/{args.task}/{args.dataset}"):
           os.makedirs(f"./result/{args.model}/{args.task}/{args.dataset}")
           
        with torch.no_grad():
            for ii, d in enumerate(test_loader):
                print(ii)
                if ii <= 1:
                    print(d['input_tokens'][0])
                input_ids = d["input_ids"].to(args.device)
                attention_mask = d["attention_mask"].to(args.device)
                output = LLM.generate_cls(input_ids=input_ids, 
                                             attention_mask=attention_mask, 
                                             max_new_tokens=1)
                print(output)
                rows.extend(
                    zip(d['index'],
                        d['input_tokens'],
                        d["labels"],
                        output,
                    )
                )
                result_df = format_eval_output(rows)  
                # result_df.to_csv(f"./result/{args.model}/{args.task}/{args.dataset}/{args.mode}_{args.train_idx}_pre_tmp.csv", index=False)
        result_df = format_eval_output(rows)        
        result_df.to_csv(f"./result/{args.model}/{args.task}/{args.dataset}/{args.mode}_{args.train_idx}_pre.csv", index=False)
        if args.write_to_train:
            print(result_df['outputs'].value_counts())
            # write prior predictions to train csv
            train_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/train_{args.train_idx}.csv")
            for i in range(len(train_df)):
                assert train_df['label'][i] == result_df['label'][i]
            train_df['prediction'] = result_df['outputs']
            train_df.to_csv(f"./dataset/{task_name}/{dataset_name}/train_{args.train_idx}.csv", index=False)
