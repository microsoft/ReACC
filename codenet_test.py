# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
import logging
import os
import random
import re
import json
import pickle
from collections import Counter
import numpy as np
import torch
from itertools import cycle
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForMaskedLM, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)

at_K = 100

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_ids,
                 index,
                 label

    ):
        self.code_ids = code_ids
        self.index= index
        self.label = label


class CodeWithDocNoRepDataset(Dataset):
    def __init__(self, tokenizer, args, file_type, cut_ratio=0.0):
        self.tokenizer = tokenizer
        self.args = args
        data_file = os.path.join(args.data_dir, f"{file_type}.jsonl")

        if args.lang == "java":
            from process_java import processor
        elif args.lang == "python":
            from process_python import processor
        self.proc = processor(args.lang, remove_comments=False)
        # self.proc.load_names(args.vars_dir)

        #load index
        logger.info(f"Creating features from {data_file}")


        self.examples = []
        lines = open(data_file).readlines()
        for i,line in enumerate(lines):
            content = json.loads(line)
            self.proc.update(content["func"])
            code = self.proc.untokenize(cut_ratio=cut_ratio, fix_cut_pos=True)
            self.proc.update(self.proc.convert_to_normal(code))
            api_seq = self.proc.get_api_seq()
            token_id = self.encode_v3(code, api_seq)
            self.examples.append(InputFeatures(token_id, content["index"], int(content["label"])))
        logger.info(f"loaded {len(self.examples)} data")

        self.label_examples={}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def encode_v3(self, code, api_seq):
        code_tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(code) + \
                      [self.tokenizer.sep_token] + self.tokenizer.tokenize(" ".join(api_seq)) + [self.tokenizer.sep_token]
        code_tokens = code_tokens[:self.args.block_size]
        code_ids = self.tokenizer.convert_tokens_to_ids(code_tokens)
        return code_ids
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].code_ids), self.examples[i].index

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def my_collect_fn(sequences, batch_first=True, padding_value=1):
    inputs1 = []
    inputs2 = []
    for (x1, x2) in sequences:
        inputs1.append(x1)
        inputs2.append(x2)
    return (
        pad_sequence(inputs1, batch_first, padding_value),
        inputs2
    )


def eval_bm25_beir(args, tokenizer, file_name, candidate_file_name, cut=False):
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.search.lexical import BM25Search as BM25

    if args.lang == "java":
        from process_java import processor
    elif args.lang == "python":
        from process_python import processor
    proc = processor(args.lang, remove_comments=False)

    idx2label = {}
    label2num = Counter()
    lines = open(os.path.join(args.data_dir, f"{candidate_file_name}.jsonl")).readlines()
    corpus = {}
    for i,line in enumerate(tqdm(lines)):
        content = json.loads(line)
        # proc.update(content["func"])
        # code = proc.untokenize()
        code = content["func"]
        idx2label[content["index"]] = content["label"]
        label2num[content["label"]] += 1
        corpus[content["index"]] = {"text": code}

    lines = open(os.path.join(args.data_dir, f"{file_name}.jsonl")).readlines()
    queries = {}
    qrels = {}
    for i,line in enumerate(tqdm(lines)):
        content = json.loads(line)
        ori_code_tokens = content["func"].split()
        if cut:
            code = " ".join(ori_code_tokens[:len(ori_code_tokens)//2])
        else:
            code = " ".join(ori_code_tokens)
        # proc.update(content["func"])
        # code = proc.untokenize(cut_ratio=1.0 if cut else 0.0, fix_cut_pos=True)
        queries[content["index"]] = code
        qrels[content["index"]] = {content["index"]: 1}

    model = BM25(index_name="codenet", hostname="http://localhost:9200", initialize=True)
    retriever = EvaluateRetrieval(model, k_values=[at_K+1])
    scores = retriever.retrieve(corpus, queries)

    # pickle.dump(scores, open(os.path.join(args.data_dir, "bm25_scores.pkl"), "wb"))

    MAP = []
    PREC = 0.0
    for idx, v in tqdm(scores.items()):
        v = sorted(v.items(), key=lambda x:-x[1])
        label = idx2label[idx]
        div = min(at_K, label2num[label])
        Avep = []
        cont = 0
        for i, (_id, score) in enumerate(v):
            if i - cont >= at_K:
                break
            if _id == idx:
                cont += 1
                continue
            if idx2label[_id] == label:
                Avep.append((len(Avep)+1)/(i+1-cont))
                if i - cont == 0:
                    PREC += 1.0
        MAP.append(sum(Avep)/div)

    result = {
        "eval_map":float(np.mean(MAP)),
        "eval_prec":float(PREC/len(MAP))
    }
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def evaluate(args, model, tokenizer, file_name, candidate_file_name, cut):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    query_dataset = CodeWithDocNoRepDataset(tokenizer, args, file_name, cut_ratio=1.0 if cut else 0.0)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, collate_fn=partial(my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id), num_workers=4)
    
    candidate_dataset = CodeWithDocNoRepDataset(tokenizer, args, candidate_file_name)
    candidate_sampler = SequentialSampler(candidate_dataset)
    candidate_dataloader = DataLoader(candidate_dataset, sampler=candidate_sampler, batch_size=args.eval_batch_size, collate_fn=partial(my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id), num_workers=4)    
    

    idx2label = {}
    label2num = Counter()
    lines = open(os.path.join(args.data_dir, f"{candidate_file_name}.jsonl")).readlines()
    corpus = {}
    for i,line in enumerate(tqdm(lines)):
        content = json.loads(line)
        idx2label[content["index"]] = content["label"]
        label2num[int(content["label"])] += 1

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num Query = %d", len(query_dataset))
    logger.info("  Num Candidate = %d", len(candidate_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.to(args.device)
    
    model.eval()
    query_vecs = [] 
    query_indexs = []
    candidate_vecs = []
    candidate_indexs = []

    for batch in tqdm(query_dataloader, total=len(query_dataloader)):  
        code_inputs = batch[0].to(args.device)
        index = batch[1]
        with torch.no_grad():
            attn_mask = torch.tensor(code_inputs.clone().detach() != tokenizer.pad_token_id, dtype=torch.uint8, device=args.device)
            code_vec = model(code_inputs, attention_mask=attn_mask)[0]
            code_vec = torch.nn.functional.normalize(code_vec[:, 0, :], dim=1)
            query_vecs.append(code_vec.cpu().numpy()) 
            query_indexs.extend(index)
            
    for batch in tqdm(candidate_dataloader,total=len(candidate_dataloader)):  
        code_inputs = batch[0].to(args.device)
        index = batch[1]
        with torch.no_grad():
            attn_mask = torch.tensor(code_inputs.clone().detach() != tokenizer.pad_token_id, dtype=torch.uint8, device=args.device)
            code_vec = model(code_inputs, attention_mask=attn_mask)[0]
            if args.num_vec > 0:
                code_vec = torch.nn.functional.normalize(code_vec[:, :args.num_vec, :], dim=2)
            else:
                code_vec = torch.nn.functional.normalize(code_vec[:, 0, :], dim=1)
            candidate_vecs.append(code_vec.cpu().numpy()) 
            candidate_indexs.extend(index)
            
    model.train() 

    query_vecs = np.concatenate(query_vecs, 0)
    candidate_vecs = np.concatenate(candidate_vecs, 0)
    query_labels = [idx2label[x] for x in query_indexs]
    candidate_labels = [idx2label[x] for x in candidate_indexs]
    
    if args.num_vec > 0:
        scores=np.einsum('nd,mvd->nmv', query_vecs, candidate_vecs).max(-1)
    else:
        scores=np.matmul(query_vecs, candidate_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

    MAP = []
    MAP_at_K = []
    PREC = 0.0
    for i in tqdm(range(scores.shape[0]), total=scores.shape[0]):
        cont = 0
        label = int(query_labels[i])
        div = min(at_K, label2num[label])
        query_index = query_indexs[i]
        Avep = []
        for j,index in enumerate(list(sort_ids[i])):
            if query_index == candidate_indexs[index]:
                cont += 1
                continue
            if j - cont == at_K:
                MAP_at_K.append(sum(Avep)/div)
            if int(candidate_labels[index]) == label:
                Avep.append((len(Avep)+1)/(j+1-cont))
                if j - cont == 0:
                    PREC += 1.0
        if len(Avep) > 0:
            MAP.append(sum(Avep)/len(Avep))
        else:
            MAP.append(0.0)
        
    result = {
        "Data size":len(MAP),
        "eval_map":float(np.mean(MAP)),
        f"eval_map_at_{at_K}":float(np.mean(MAP_at_K)),
        "eval_prec":float(PREC/len(MAP))
    }
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--pretrained_dir", default=None, type=str,
                        help="The directory where the trained model and tokenizer are saved.")
    parser.add_argument("--lang", default="python", type=str, 
                        help="Language of dataset")

    parser.add_argument("--cut", action='store_true',
                        help="Ratio of replaced variables")
    parser.add_argument('--num_vec', type=int, default=-1,
                        help="number of vectors")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(args)

    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_dir)
    model = RobertaModel.from_pretrained(args.pretrained_dir, add_pooling_layer=False)


    evaluate(args, model, tokenizer, args.lang, args.lang, args.cut)
    # eval_bm25_beir(args, tokenizer, "java", "java", cut=True)

if __name__ == "__main__":
    main()

