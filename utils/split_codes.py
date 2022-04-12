# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This script is used for splitting long codes into small chunks with the same length (300 by default) in search base.

import os
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file_name', '-i', required=True, help="filename of the codes, should be in txt format")
    parser.add_argument('--length', '-l', type=int, default=300, help="length of the chunk")
    args = parser.parse_args()

    lines = open(args.file_name, "r").readlines()
    wf = open(args.file_name.split("/")[-1].split(".")[0]+"_split.txt", "w")
    nexts = []
    cnt = 0
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) <= args.length:
            wf.write(" ".join(tokens)+"\n")
            nexts.append(cnt)
            cnt += 1
        else:
            for i in range(0, len(tokens), args.length):
                wf.write(" ".join(tokens[i:i+args.length])+"\n")
                nexts.append(cnt+1)
                cnt += 1
            nexts[-1] -= 1
    wf.close()
    pickle.dump(nexts, open(args.file_name.split("/")[-1].split(".")[0]+"_split_nexts.pkl", "wb"))

if __name__ == "__main__":
    main()


