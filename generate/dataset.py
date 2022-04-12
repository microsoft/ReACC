# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import gc
import shutil
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


class RelineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924, load_file=None, search_res=None):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")
        with open(datafile) as f:
            datas = f.readlines()

        if load_file is not None:
            id2code = {}
            lines = open(os.path.join(args.data_dir, load_file+".txt")).readlines()
            for i,line in enumerate(tqdm(lines)):
                id2code[i] = line.strip()

            search_results = pickle.load(open(search_res, "rb"))
            try:
                nexts = pickle.load(open(os.path.join(args.data_dir, load_file+"_nexts.pkl"), "rb"))
            except Exception:
                nexts = [i for i in range(len(lines))]

        length = len(datas)
        logger.info("Data size: %d"%(length))
        self.inputs = []
        self.gts = []
        for i,data in enumerate(datas):
            if i % 1000 == 0:
                logger.info(f"Encoded {i}/{length} data")
            data = json.loads(data.strip())
            if load_file is not None:
                try:
                    cand_id = search_results[data["id"]]
                    cand = id2code[cand_id]
                    if nexts[cand_id] != cand_id:
                        cand += id2code[nexts[cand_id]]
                    cand = tokenizer.encode(cand)
                except:
                    cand = []
            else:
                cand = []
            self.inputs.append((cand + tokenizer.encode(data["input"]))[-block_size:])
            self.gts.append(data["gt"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]

class PPLDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=1024, load_file=None, search_res=None):
        datafile = os.path.join(args.data_dir, f"{file_type}.txt")
        with open(datafile) as f:
            datas = f.readlines()

        if load_file is not None:
            id2code = {}
            lines = open(os.path.join(args.data_dir, load_file+".txt")).readlines()
            for i,line in enumerate(tqdm(lines)):
                id2code[i] = line.strip()

            search_results = pickle.load(open(search_res, "rb"))
            try:
                nexts = pickle.load(open(os.path.join(args.data_dir, load_file+"_nexts.pkl"), "rb"))
            except Exception:
                nexts = [i for i in range(len(lines))]

        length = len(datas)
        logger.info("Data size: %d"%(length))
        self.inputs = []
        self.token_labels = []
        for i,data in enumerate(tqdm(datas)):
            if i % 1000 == 0:
                logger.info(f"Encoded {i}/{length} data")
            tokens = data.strip().split(" ")
            if len(tokens) < 200:
                cut = len(tokens)//2
            else:
                cut = 100

            if load_file is not None:
                try:
                    if i in search_results:
                        cand_id = search_results[i]
                    else:
                        cand_id = search_results[str(i)]
                    cand = id2code[cand_id]
                    if nexts[cand_id] != cand_id:
                        cand += id2code[nexts[cand_id]]
                    cand = tokenizer.encode(cand)
                except:
                    # print("OK")
                    cand = []
            else:
                cand = []
            
            x1 = tokenizer.encode(" ".join(tokens[:cut]))[-block_size:]
            self.inputs.append(x1)
            self.token_labels.append([2]*len(x1))

            pre_id = cand + tokenizer.encode(" ".join(tokens[:cut]))
            x2_0 = tokenizer.encode(" ".join(tokens[cut:]))[:block_size * 3 // 4]
            x2 = (pre_id + x2_0)[-block_size:]
            self.inputs.append(x2)
            self.token_labels.append([1]*(len(x2)-len(x2_0)) + [2]*len(x2_0))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item])
