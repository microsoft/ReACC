# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script is used to generate the embedding vectors for the given dataset.

import argparse
import logging
import os
import random
import re
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from itertools import cycle
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

class InferDataset(Dataset):
    def __init__(self, tokenizer, args, api=True):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()
        self.tokenizer = tokenizer
        self.args = args
        self.api = api
        data_file = args.data_path


        if args.lang == "java":
            from process_java import processor
        elif args.lang == "python":
            from process_python import processor
        self.proc = processor(args.lang, remove_comments=False)

        logger.info(f"Creating features from {data_file}")
        data_format = data_file.split(".")[-1]

        self.data = []
        self.idx = []
        n = 0
        with open(data_file) as f:
            for _ in f:
                n += 1
        # n = 100000
        st = n//world_size*local_rank
        ed = n//world_size*(local_rank+1)
        logger.warning(f"device {local_rank} will load {ed-st} data line from {st} to {ed}")
        with open(data_file) as f:
            for i,line in enumerate(f):
                if i >= st and i < ed:
                    if (i-st) % 100000 == 0:
                        logger.info(f"device {local_rank} created {i-st}/{ed-st} train data")
                    if "json" in data_format:
                        content = json.loads(line)
                        self.data.append(self.convert_cxg_format_to_normal(content["input"]))
                        self.idx.append(content["id"])
                    else:   # txt
                        self.data.append(self.convert_cxg_format_to_normal(line.strip()))
                        self.idx.append(i)
        logger.warning(f"device {local_rank} loaded {len(self.data)} train data from {st} to {ed}")

    def convert_cxg_format_to_normal(self, code):
        if code.startswith("<s>"):
            code = code.lstrip("<s>")
        if code.endswith("</s>"):
            code = code.rstrip("</s>")
        code = code.replace("<EOL>", "\n")
        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
        return code

    def encode(self, code, api_seq):
        if self.api:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + \
                          [self.tokenizer.sep_token] + self.tokenizer.tokenize(" ".join(api_seq)) + [self.tokenizer.sep_token]
        else:
            code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + [self.tokenizer.sep_token]
        code_tokens = code_tokens[:self.args.block_size]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return code_ids

    def process(self, code):
        self.proc.update(code)
        api_seq = self.proc.get_api_seq()
        code = self.proc.untokenize(cut_ratio=0.0)
        token_id = self.encode(code, api_seq)
        return token_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return torch.tensor(self.process(self.data[item])), torch.tensor([self.idx[item]])  



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def my_collect_fn(sequences, batch_first=True, padding_value=1):
    inputs = []
    inputs1 = []
    for (x, x1) in sequences:
        inputs.append(x)
        inputs1.append(x1)
    return (
        pad_sequence(inputs, batch_first, padding_value),
        pad_sequence(inputs1, batch_first, padding_value),
    )

def inference(args, tokenizer, model, save_name, api=False):
    #build dataloader
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    dataset = InferDataset(tokenizer, args,  api=api)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=partial(my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id), num_workers=4)
    
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                        output_device=args.local_rank%args.gpu_per_node,
                                                        find_unused_parameters=True)

    # Eval!
    logger.info("***** Running Inference *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    steps = 0
    n_vec = max(0, args.num_vec)
    saved = {}
    for batch in dataloader:
        with torch.no_grad():
            (inputs1, inputs2) = batch
            inputs1 = inputs1.to(args.device)
            attn_mask1 = torch.tensor(inputs1.clone().detach() != tokenizer.pad_token_id, dtype=torch.uint8, device=args.device)
            outputs = model(inputs1, attention_mask=attn_mask1)[0]
            if n_vec > 0:
                outputs = nn.functional.normalize(outputs[:, :n_vec, :], dim=2)
            else:
                outputs = nn.functional.normalize(outputs[:, 0, :], dim=1)
            outputs = outputs.detach().to("cpu").numpy()
            idxs = inputs2.numpy()
        for i in range(outputs.shape[0]):
            saved[idxs[i][0]] = outputs[i]
        steps += 1
        if steps % args.logging_steps == 0:
            logger.info(f"Inferenced {steps} steps")
    
    if args.local_rank != -1:
        pickle.dump(saved, open(save_name+f"_{args.local_rank}.pkl", "wb"))
    else:
        pickle.dump(saved, open(save_name+".pkl", "wb"))

def merge(args, num, save_name):
    saved = {}
    for i in range(num):
        saved.update(pickle.load(open(save_name+f"_{i}.pkl", "rb")))
        os.remove(save_name+f"_{i}.pkl")
    pickle.dump(saved, open(save_name+".pkl", "wb"))




def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_path", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--save_name", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="Language of the dataset.")
    parser.add_argument("--pretrained_dir", default=None, type=str,
                        help="The directory where the trained model and tokenizer are saved.")

    parser.add_argument("--cut_ratio", type=float, default=0.5,
                        help="Ratio of replaced variables")
    parser.add_argument('--num_vec', type=int, default=-1,
                        help="number of vectors")

    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=0,
                        help="node index if multi-node running")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")

    args = parser.parse_args()

    logger.warning("local_rank: %d, node_index: %d, gpu_per_node: %d"%(args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, world size: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), world_size)

    # Set seed
    set_seed(args)

    args.start_epoch = 0
    args.start_step = 0

    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_dir)
    model = RobertaModel.from_pretrained(args.pretrained_dir, add_pooling_layer=False)

    inference(args, tokenizer, model, args.save_name, api=True)
    logger.info(f"device {args.local_rank} finished")

    if args.local_rank != -1:
        torch.distributed.barrier()
        if args.local_rank in [-1, 0]:
            import time
            time.sleep(10)
            merge(args, world_size, save_name=args.save_name)


if __name__ == "__main__":
    main()

