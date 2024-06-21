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
    default="Llama-2-13b-chat-hf",  
    help="LLM directory / name")
parser.add_argument(
    "--task",
    type=str,
    choices=["ATSC", "SC", "EMO", "Irony", "Stance", "NLI"],
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
    "--num_examples", type=int, default=4, help="Number of in-context examples"
)
parser.add_argument(
    "--seed", type=int, default=123, help="Random seed"
)
parser.add_argument(
    "--run_pre", action="store_true", default=False, help="Run zero-shot gpt"
)
parser.add_argument(
    "--run_zero", action="store_true", default=False, help="Run zero-shot gpt"
)
parser.add_argument(
    "--run_icl", action="store_true", default=False, help="Run ICL baseline"
)
parser.add_argument(
    "--icl_mode", type=str, default="random_each"
)
parser.add_argument(
    "--run_ficl", action="store_true", default=False, help="Run SuperICL"
)
parser.add_argument(
    "--batch_size", type=int, default=24, help="Batch size for Inference"
)
parser.add_argument(
    "--load_bit", type=str, default="fp16"
)
parser.add_argument(
    "--device", type=str, default="cuda:0"
)
parser.add_argument(
    "--max_len", type=int, default=720
)
parser.add_argument(
    "--wrong_exps", type=int, default=2
)
parser.add_argument(
    "--balance", type=int, default=0
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
    print(f"ICL: {args.icl_mode}")
    print(f"Number of training data: {args.train_examples}")
    print(f"Number of Demos: {args.num_examples}")
    print("Loading Tokenizer")
    tokenizer = Tokenizer4LLM(args.max_len, args.model, llm=True)
    dataset_name = args.dataset
    task_name = args.task
    # Create a list to store the threads and their results
    
    # load dataset
    train_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/train_{args.train_idx}.csv")
    if "GPT-J" in args.model:
        train_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/train_{args.train_idx}_gptj.csv")
    train_df = train_df[:args.train_examples]  # number of training set examples e.g. top500
    assert len(train_df) == args.train_examples
    # confirm the data type
    train_df['label'] = train_df['label'].astype(str)
    train_df['prediction'] = train_df['prediction'].astype(str)
    assert train_df.index.tolist() == list(range(args.train_examples))
    test_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/test.csv")
    # dev_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/dev.csv")
    
    # Select icl examples
    if args.icl_mode == "random_each":
        example_ids = readJson(f"./ICL_examples/{task_name}/{dataset_name}/random_each_{args.train_idx}.json")
    elif args.icl_mode == "bm25":
        example_ids = readJson(f"./ICL_examples/{task_name}/{dataset_name}/bm25_{args.train_idx}.json")
    elif args.icl_mode == "kmeans":
        example_ids = readJson(f"./ICL_examples/{task_name}/{dataset_name}/kmeans_{args.train_idx}.json")
    elif args.icl_mode == "sbert":
        example_ids = readJson(f"./ICL_examples/{task_name}/{dataset_name}/sbert_{args.train_idx}.json")
    elif args.icl_mode == "mmr":
        example_ids = readJson(f"./ICL_examples/{task_name}/{dataset_name}/mmr_{args.train_idx}.json")
    # Filter by current training data
    train_df_index = train_df.index.tolist()
    new_example_ids = dict()
    for k, v in example_ids.items():
        examples = [int(i) for i in v]
        new_example_ids[int(k)] = [i for i in examples if i in train_df_index]
    example_ids = new_example_ids
    if not os.path.exists(f"./result/{args.model}/{args.task}/{args.dataset}"):
        os.makedirs(f"./result/{args.model}/{args.task}/{args.dataset}")
    if args.run_icl:
        prompt = icl_instruction_prompt[dataset_name]
        all_labels = label_space[dataset_name]
        TestDataset = ClassificationDataset(
                        test_df, args, prompt,
                        get_input_template, 
                        all_labels,
                        tokenizer=tokenizer, 
                        icl_mode=args.icl_mode, 
                        examples=train_df, 
                        example_ids=example_ids
                    )
        
        test_loader = DataLoader(
                        TestDataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False) 
        
        print("Loading LLM")
        LLM = LLMgenerate(args, tokenizer, all_labels=all_labels)
        # Results
        rows = []
        print("Start Generation with the LLM")
        with torch.no_grad():
            for ii, d in enumerate(test_loader):
                print(ii)
                if ii <= 1:
                    print(d['input_tokens'][0])
                input_ids = d["input_ids"].to(args.device)
                attention_mask = d["attention_mask"].to(args.device)
                output = LLM.generate_cls(input_ids=input_ids, 
                                             attention_mask=attention_mask, max_new_tokens=1)
                rows.extend(
                    zip(d['index'],
                        d['input_tokens'],
                        d["labels"],
                        output,
                    )
                )
                print(output)
        result_df = format_eval_output(rows)  
        
        if not os.path.exists(f"./result/{args.model}/{args.task}/{args.dataset}"):
            os.makedirs(f"./result/{args.model}/{args.task}/{args.dataset}")     
        
        result_df.to_csv(f"./result/{args.model}/{args.task}/{args.dataset}/train{args.train_idx}_{args.icl_mode}_{args.num_examples}.csv", index=False)

    elif args.run_ficl:
        # test_df = pd.read_csv(f"./dataset/{task_name}/{dataset_name}/test_pred.csv")
        prompt = icl_instruction_prompt[dataset_name]
        all_labels = label_space[dataset_name]
        TestDataset = ClassificationDataset(
                        test_df, args, prompt,
                        get_input_template, 
                        all_labels,
                        tokenizer=tokenizer, 
                        icl_mode=args.icl_mode, 
                        examples=train_df, 
                        example_ids=example_ids,
                        feedback_prompt=feedback_prompt,
                    )
        
        test_loader = DataLoader(
                        TestDataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False
                    )  
        print("Loading LLM")
        LLM = LLMgenerate(args, tokenizer, all_labels=all_labels)
        # Results
        rows = []
        print("Start Generation with the LLM")
        with torch.no_grad():
            for ii, d in enumerate(test_loader):
                print(ii)
                if ii <= 1:
                    print(d['input_tokens'][0])
                input_ids = d["input_ids"].to(args.device)
                attention_mask = d["attention_mask"].to(args.device)
                output = LLM.generate_cls(input_ids=input_ids, 
                                             attention_mask=attention_mask, max_new_tokens=1)
                rows.extend(
                    zip(d['index'],
                        d['input_tokens'],
                        d["labels"],
                        output,
                    )
                )
                print(output)
        result_df = format_eval_output(rows)        
        # _balance_{args.balance}
        
        if not os.path.exists(f"./result/{args.model}/{args.task}/{args.dataset}"):
            os.makedirs(f"./result/{args.model}/{args.task}/{args.dataset}")   
            
        result_df.to_csv(f"./result/{args.model}/{args.task}/{args.dataset}/train{args.train_idx}_{args.icl_mode}_{args.num_examples}_ficl.csv", index=False)