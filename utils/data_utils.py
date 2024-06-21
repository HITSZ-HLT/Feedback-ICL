# -*- coding: utf-8 -*-
# file: data_utils.py

import os
import ast
import pickle
import numpy as np
import torch
import random
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def reorder_list_by_new_order(original_list, new_order):
    # 使用新顺序中的位置重新排列列表
    reordered_list = [original_list[int(i)] for i in new_order]
    return reordered_list
    
class Tokenizer4LLM:
    def __init__(self, max_seq_len, pretrained_model, llm=True):
        if llm:
            print(" Loading Tokenizer for LLMs ! ")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, padding_side='left', trust_remote_code=True)
            if 'GPT' in pretrained_model or "llama" in pretrained_model:
                self.tokenizer.pad_token = self.tokenizer.eos_token 
            self.max_seq_len = max_seq_len
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.max_seq_len = max_seq_len

    def input4llm(self, text):
        # simple tokenization (with left padding) for Large Language Models
        encoding = self.tokenizer(
                        text,
                        max_length = self.max_seq_len,
                        padding="max_length",
                        return_tensors="pt"
                    )

        return encoding

class ClassificationDataset(Dataset):
    # Dataset for In-Context Learning
    # both for Openai API and Local LLMs
    def __init__(self, data, cfg, instruct_prompt, get_template, all_labels=None, tokenizer=None, icl_mode=None, 
                    examples=None, example_ids=None, feedback_prompt=None, thinking_prompt=None):
        self.cfg = cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.dataset_name = cfg.dataset
        self.prompt = instruct_prompt
        self.tokenizer = tokenizer # no need for openai api model
        self.get_input_template = get_template
        self.icl_mode = icl_mode  # ICL mode
        if icl_mode is not None:
            self.n_example = cfg.num_examples
        self.examples = examples  # Train Dataset, from which is used to select ICL examples  
        self.example_ids = example_ids  # Fixed ICL example Index (might be None)
        self.data = data # test_data
        self.all_labels = all_labels
        self.feedback_prompt = feedback_prompt
        self.thinking_prompt = thinking_prompt

    def __getitem__(self, index):
        data = dict(self.data.iloc[index])
        tokens = data['sentence']
        label = f"{data['label']}"
        examples_with_template = []        
        # input tokens ()
        if self.cfg.run_pre:
            input_prompt = self.prompt
            examples = []
            while True:
                # 随机产生一个整数
                j = random.randint(0, len(self.data)-1)
                if j in examples or j == index:
                    continue
                examples.append(j)
                examples_with_template.append(dict(self.data.iloc[j]))   
                if len(examples) == 4:
                    break
            input_prompt = self.prompt
            for example in examples_with_template:
                input_prompt += self.get_input_template(example, self.dataset_name) + f"Label: {example['label']}"+"\n\n"
            input_tokens = input_prompt + self.get_input_template(data, self.dataset_name) + "Label: "
        elif self.cfg.run_icl:
            # get ICL examples
            example_id = self.example_ids[index] # get corresponding example id
            # get TopK examples by id in the train dataset
            examples = []
            for i in range(self.n_example):
                for j in example_id:
                    if j in examples:
                        continue
                    examples.append(j)
                    examples_with_template.append(dict(self.examples.iloc[j]))   
                    break     
            input_prompt = self.prompt
            for example in examples_with_template:
                input_prompt += self.get_input_template(example, self.dataset_name) + f"Label: {example['label']}"+"\n\n"
            input_tokens = input_prompt + \
                self.get_input_template(data, self.dataset_name) + "Label: "  
        elif self.cfg.run_ficl:
            example_id = self.example_ids[index] # get corresponding example id
            # get TopK examples by id in the train dataset
            examples = []
            preds, truth = [], []
            for i in range(self.n_example):
                if i < self.cfg.wrong_exps:
                    for num, j in enumerate(example_id):
                        if j in examples:
                            continue
                        if self.cfg.icl_mode == "kmeans" and self.cfg.num_examples == 4:
                            # 保证4-shot样本位于不同的cluster
                            for e in examples:
                                if num % 4 == examples.index(e) % 4:
                                    continue
                        if self.examples.iloc[j]['label'] != self.examples.iloc[j]['prediction']:
                            examples_with_template.append(dict(self.examples.iloc[j]))
                            examples.append(j)
                            preds.append(self.examples.iloc[j]['prediction'])
                            truth.append(self.examples.iloc[j]['label'])
                            break
                else:
                    for j in example_id:
                        if j in examples:
                            continue
                        if self.examples.iloc[j]['label'] == self.examples.iloc[j]['prediction']:
                            examples_with_template.append(dict(self.examples.iloc[j]))
                            examples.append(j)
                            preds.append(self.examples.iloc[j]['label'])
                            truth.append(self.examples.iloc[j]['label'])
                            break
            input_prompt = self.prompt
            
            # assert len(examples_with_template) == self.n_example
            
            for ii, example in enumerate(examples_with_template):
                if preds[ii] != truth[ii]:
                    input_prompt += f"{self.get_input_template(example, self.dataset_name)}" + f"Prediction: {preds[ii]}\n"\
                         + f"Correct Label: {example['label']}\n" + self.feedback_prompt["wrong"]
                else:
                    input_prompt += f"{self.get_input_template(example, self.dataset_name)}" + f"Prediction: {example['label']}\n"\
                         + f"Correct Label: {example['label']}\n" + self.feedback_prompt["correct"]
            # pred = data['prediction']
            input_tokens = input_prompt + self.get_input_template(data, self.dataset_name) + "Correct Label: " #   self.thinking_prompt 
            
        if self.tokenizer:
            # for local LLMs
            encoding = self.tokenizer.input4llm(input_tokens)
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return  {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'index': str(index),
                    'tokens': tokens,
                    'input_tokens': input_tokens,
                    'labels': label,
            }
        else:
            # for LLMs using API (no need for tokenizer)
            return  {
                    'index': str(index),
                    'tokens': tokens,
                    'input_tokens': input_tokens,
                    'labels': label,
            }

    def __len__(self):
        return len(self.data) # length of the data dict