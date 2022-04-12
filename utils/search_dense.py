# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import faiss


def search(index_file, query_file, save_name):
    index_data = pickle.load(open(index_file, "rb"))
    query_data = pickle.load(open(query_file, "rb"))
    ids = []
    indexs = []
    id2n = {}
    for i, (idx, vec) in enumerate(index_data.items()):
        ids.append(idx)
        indexs.append(vec)
        id2n[idx] = i
    queries = []
    idxq = []
    for idx, vec in query_data.items():
        queries.append(vec)
        idxq.append(idx)
    ids = np.array(ids)
    indexs = np.array(indexs)
    queries = np.array(queries)

    # build faiss index
    d = 768
    k = 101
    index = faiss.IndexFlatIP(d)
    assert index.is_trained

    index_id = faiss.IndexIDMap(index)
    index_id.add_with_ids(indexs, ids)

    res = {}
    D, I = index_id.search(queries, k)
    for i, (sd, si) in enumerate(zip(D, I)):
        res[str(idxq[i])] = {}
        for pd, pi in zip(sd, si):
            res[str(idxq[i])][str(pi)] = pd
            # if pi != idxq[i]:
            #     res[str(idxq[i])][str(pi)] = pd

    pickle.dump(res, open(save_name, "wb"))

def search_multi(index_file, query_file, save_name):
    index_data = pickle.load(open(index_file, "rb"))
    query_data = pickle.load(open(query_file, "rb"))
    ids = []
    indexs = []
    for i, (idx, vecs) in enumerate(index_data.items()):
        for vec in vecs:
            ids.append(idx)
            indexs.append(vec)
    queries = []
    idxq = []
    for idx, vec in query_data.items():
        queries.append(vec[0])
        idxq.append(idx)
    ids = np.array(ids)
    indexs = np.array(indexs)
    queries = np.array(queries)
    print(indexs.shape, queries.shape)

    # build faiss index
    d = 768
    k = 100
    index = faiss.IndexFlatIP(d)
    assert index.is_trained

    index_id = faiss.IndexIDMap(index)
    index_id.add_with_ids(indexs, ids)

    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index_id)

    res = {}
    D, I = gpu_index.search(queries, k)
    for i, (sd, si) in enumerate(zip(D, I)):
        res[str(idxq[i])] = {}
        for pd, pi in zip(sd, si):
            if str(pi) not in res[str(idxq[i])]:
                res[str(idxq[i])][str(pi)] = pd
                if len(res[str(idxq[i])]) > 100:
                    break

    pickle.dump(res, open(save_name, "wb"))

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--index_file', '-i', required=True, help="filename of index embeddings saved")
    parser.add_argument('--query_file', '-q', required=True, help="file containing query embeddings")
    parser.add_argument('--save_name', '-o', required=True, help="save file name")
    parser.add_argument("--multi", action='store_true', help="set true if one query/doc has multi embeddings")
    args = parser.parse_args()

    if args.multi:
        search_multi(args.index_file, args.query_file, args.save_name)
    else:
        search(args.index_file, args.query_file, args.save_name)

if __name__ == "__main__":
    main()

