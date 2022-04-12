# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import argparse
import json
from tqdm import tqdm
import pickle
import random
import csv
import time

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

def build(corpus_file, query_file, temp_dir):
    # corpus and query file have to be plain text files
    print("Building bm25 corpus")
    corpus = []
    queries = []
    ids = []
    datas = open(corpus_file).readlines()
    lines = open(query_file).readlines()
    try:
        os.mkdir(temp_dir)
    except FileExistsError:
        pass
    fidx = open(os.path.join(temp_dir, "corpus.jsonl"), "w")
    fq = open(os.path.join(temp_dir, "query.jsonl"), "w")
    fr = open(os.path.join(temp_dir, "res.tsv"), "w")
    csv_fr = csv.writer(fr, delimiter='\t')
    fr.write("q\td\t\s\n")
    for i,line in enumerate(tqdm(datas)):
        fidx.write(json.dumps({"_id":str(i), "text":line.strip()})+"\n")
    for i,line in enumerate(tqdm(lines)):
        content = json.loads(line)
        idx = content["id"]
        csv_fr.writerow([str(idx), str(idx), 1])
        code = content["input"].strip()
        fq.write(json.dumps({"_id":str(idx), "text":code})+"\n")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--search_corpus', '-i', required=True, help="search corpus file, plain text file")
    parser.add_argument('--query_file', '-q', required=True, help="queries file, json file")
    parser.add_argument('--save_name', '-o', required=True, help="same file name")
    parser.add_argument('--temp_path', '-t', default="beir", help="temp dir to save beir-format data")
    args = parser.parse_args()

    build(args.search_corpus, args.query_file, args.temp_path)
    time.sleep(10)

    corpus, queries, qrels = GenericDataLoader(
        corpus_file=os.path.join(args.temp_path, "corpus.jsonl"), 
        query_file=os.path.join(args.temp_path, "query.jsonl"),
        qrels_file=os.path.join(args.temp_path, "res.tsv")
    ).load_custom()

    model = BM25(index_name="reacc", hostname="127.0.0.1:9200", initialize=True)
    retriever = EvaluateRetrieval(model)

    results = retriever.retrieve(corpus, queries)
    pickle.dump(results, open(args.save_name, "wb"))

if __name__ == "__main__":
    main()
