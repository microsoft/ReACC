# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import pickle
import argparse
import json
import numpy as np
from tqdm import tqdm
import random

def hybrid_scores(bm25_scores, dense_scores, alpha, beilv=100):
    # beilv: re-scaling dense score as it is percentage.
    scores = {}
    for idx, v in tqdm(dense_scores.items()):
        new_v = {}
        if idx not in bm25_scores:
            scores[idx] = v
            continue
        v2 = bm25_scores[idx]
        v_min = min(list(v.values()))
        v2_min = min(list(v2.values()))
        for _id, score in v.items():
            if _id not in v2:
                new_v[_id] = beilv * score + alpha * v2_min
            else:
                new_v[_id] = beilv * score + alpha * v2[_id]
        for _id, score in v2.items():
            if _id not in new_v:
                new_v[_id] = alpha * score + beilv * v_min
        scores[idx] = new_v
    return scores

def get_res(bm25_file, dense_file, save_file, alpha):
    if bm25_file != "":
        bm25_scores = pickle.load(open(bm25_file, "rb"))
        print("bm25 scores loaded")
    else:
        bm25_scores = {}
    if dense_file != "":
        dense_scores = pickle.load(open(dense_file, "rb"))
        print("dense scores loaded")
    else:
        dense_scores = {}

    res = {}
    if len(bm25_scores) > 0 and len(dense_scores) > 0:
        scores = hybrid_scores(bm25_scores, dense_scores, alpha, 100)
    elif len(bm25_scores) > 0:
        scores = bm25_scores
    else:
        scores = dense_scores
    for idx, v in tqdm(scores.items()):
        v = sorted(v.items(), key=lambda x:-x[1])
        # res[int(idx)] = int(v[0][0]) if v[0][0] != idx else int(v[1][0])
        res[int(idx)] = int(v[0][0])
        
    pickle.dump(res, open(save_file, "wb"))


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_name', '-o', required=True, help="save file name")
    parser.add_argument('--bm25_res', '-b', default="", help="bm25 result file")
    parser.add_argument('--dense_res', '-d', default="", help="dense result file")
    parser.add_argument("--alpha", type=float, default=1.1, help="ratio of dense score")
    args = parser.parse_args()

    get_res(args.bm25_res, args.dense_res, args.save_name, args.alpha)


if __name__ == "__main__":
    main()


