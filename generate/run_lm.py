# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import absolute_import, division, print_function

import argparse
from functools import partial
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import RelineDataset, PPLDataset
from beam import Beam

from fuzzywuzzy import fuzz
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def update_config(args, config):
    # config.n_positions = config.n_ctx = args.block_size
    config.vocab_size = args.vocab_size

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens

def my_collect_fn(sequences, batch_first=True, padding_value=1):
    inputs1 = []
    inputs2 = []
    for (x1, x2) in sequences:
        inputs1.append(x1)
        inputs2.append(x2)
    return (
        pad_sequence(inputs1, batch_first, padding_value),
        pad_sequence(inputs2, batch_first, 0),
    )

def eval_ppl(args, model, tokenizer, file_type='test', load_file="train", res_file="dense.pkl"):
    model.to(args.device)
    if load_file is None:
        dataset = PPLDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size)
    else:
        dataset = PPLDataset(
            tokenizer, args, logger, file_type=file_type, block_size=args.block_size, 
            load_file=load_file, 
            search_res=os.path.join(args.data_dir, "search_results", res_file), 
        )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, collate_fn=partial(my_collect_fn, batch_first=True, padding_value=tokenizer.pad_token_id), batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    num_tok = 0
    nb_eval_steps = 0
    model.eval()
    
    for step, (batch, token_labels) in enumerate(eval_dataloader):

        inputs = batch.to(args.device)
        attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8, device=args.device)
        loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8, device=args.device)
        with torch.no_grad():
            outputs = model(inputs, attention_mask=attn_mask)
            logits = outputs[0]
            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            all_labels =  shift_labels.view(-1)[ids]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[ids], all_labels)
            eval_loss += loss.item()*all_labels.shape[0]
            num_tok += all_labels.shape[0]
        
        if step % args.logging_steps == 0:
            logger.info(f"Test steps: {step}")
        nb_eval_steps += 1

    eval_loss = eval_loss / num_tok
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": float(perplexity)
    }

    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

def eval_line_completion(args, model, tokenizer, file_type='test', load_file=None, res_file=None, save_name=None):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommanded to use single GPU because it could not be batched.
    """

    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    if load_file is None:
        dataset = RelineDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size-100)
    else:
        dataset = RelineDataset(
            tokenizer, args, logger, file_type=file_type, block_size=args.block_size-100, 
            load_file=load_file, 
            search_res=res_file, 
        )
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    # model.zero_grad()
    model.eval()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    if args.langs == "python":
        break_ids = [tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
    preds = []
    gts = []
    edit_sim = 0.0
    em = 0.0
    for step, (inputs, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 5
            m = torch.nn.LogSoftmax(dim=-1)
            outputs = model(inputs[:, :-1])[1]
            p = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                past_hidden = tuple(tuple(xx[i:i+1, :].expand(beam_size, -1, -1, -1) for xx in x) for x in outputs)
                # past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1, -1) for x in outputs]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)
                input_ids = None
                for _ in range(100):
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    outputs = model(input_ids, past_key_values=past_hidden)
                    out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    past_hidden = tuple(tuple(xx.data.index_select(0, beam.getCurrentOrigin()) for xx in x) for x in outputs[1])
                    # past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in outputs[1]]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p]+[zero]*(100-len(p))).view(1,-1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = DecodeIds(t).strip("<EOL>").strip()
                else:
                    text = DecodeIds(t).strip("{").strip()
                preds.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0
        if step % args.logging_steps == 0:
            logger.warning(f"{step} are done!")
    
    with open(save_name, "w") as f:
        for pred_text in preds:
            f.write(pred_text+"\n")
            
    logger.warning(f"Test {len(preds)} samples")
    logger.warning(f"Edit sim: {edit_sim/len(preds)}, EM: {em/len(preds)}")


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--load_file_name", default=None, type=str,
                        help="search corpus to load")
    parser.add_argument("--search_res", default=None, type=str,
                        help="file that saves search results")
    parser.add_argument("--save_name", default=None, type=str,
                        help="file to save model predictions")

    ## Other parameters
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")

    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--eval_line", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    pool = None
    args = parser.parse_args()

    # args.output_dir = os.path.join(args.output_dir, args.dataset)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Set seed
    set_seed(args)

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(pretrained, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        logger.warning(f"Loading model from {pretrained}")
        model = model_class.from_pretrained(pretrained)
        model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        args.vocab_size = len(tokenizer)
        config = config_class.from_pretrained(args.config_dir)
        model = model_class(config)
        model.resize_token_embeddings(len(tokenizer))

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.warning(f"Model has a total of {num_params} trainable parameters")

    logger.warning("Training/evaluation parameters %s", args)
    
    # Only works on single GPU
    if args.eval_line:
        eval_line_completion(
            args, 
            model, 
            tokenizer, 
            file_type="test", 
            load_file=args.load_file_name, 
            res_file=args.search_res, 
            save_name=args.save_name
        )


if __name__ == "__main__":
    main()
