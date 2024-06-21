import os
import json
import random
import torch
import numpy as np
import pandas as pd
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

def get_logger(filename):
    '''
    return output log
    '''
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    '''
    setting random seed for reproduction & multiple runs
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def readJson(path, encoding='gbk'):
    '''
    read json file, return the dict
    '''
    with open(path,'r',encoding=encoding) as load_f:
        load_dict = json.load(load_f)
    return load_dict

def writeJson(path, new_dict, encoding='gbk'):
    '''
    write dict into the json file
    '''
    with open(path,"w",encoding=encoding) as f:
        json.dump(new_dict,f)