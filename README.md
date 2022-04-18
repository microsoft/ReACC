# ReACC

Source codes for ACL 2022 paper "[ReACC: A Retrieval-Augmented Code Completion Framework](https://arxiv.org/abs/2203.07722)" 
ReACC combines a source code retiever and an auto-regresstive language model for programming languages. 

## Dependency

- pytorch >= 1.7.0
- transformers >= 4.10.0
- tree_sitter
- faiss-gpu
- beir (for BM25)
  - elastic search

## Instructions

Here are the instructions to apply ReACC framework on the code completion task with PY150 dataset.

### 1. Pretrain a retriever

Leverage [`microsoft/reacc-py-retriever`](https://huggingface.co/microsoft/reacc-py-retriever) as a code-to-code retriever for python source codes.

### 2. Build an index for search

First, you have to prepare a codebase for retrieving. It is recommended to split each file/function into small chunks. (refer to `utils/split_codes.py`). Then run the command to get representations of all the codes in search corpus.

```bash
python -m torch.distributed.launch --nproc_per_node=${PER_NODE_GPU} infer.py \
        --data_path=data/train_split.txt \
        --save_name=save_vec \
        --lang=python \
        --pretrained_dir=microsoft/reacc-py-retriever \
        --num_vec=8 \
        --block_size=512 \
        --gpu_per_node ${PER_NODE_GPU} \
        --logging_steps=100 
```

You can modify the `InferDataset` in `infer.py` to fit your own dataset. Our dataset is formated as a jsonl file, where each line is like
```json
{
        "code": "def function()",
        "id": 0
}
```
or a plain text file, in which each line is a code snippet.

### 3. Retrieve step

ReACC is a two-stage framework. The first stage is to retrieve the similar codes given a query. As the test set is fixed, we retrieve all the similar codes of the queries in test set in advance. **It would be better to merge step 3 into step 4.**

First, get the representations of test queries like in step 2. Then run the script `utils/search_dense.py` to sort the similarity and get the most similar codes.

If you would like to use BM25 algorithm to retrieve similar codes, run the script `utils/search_bm25.py`.

At last, run `utils/get_res.py` to get the most similar code based on bm25 results, or dense retrieval results, or both.

### 4. Generation step

Please download PY150 dataset first and use preprocess scripts in [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-token). And follow CodeXGLUE to fine-tune a model on it, like CodeGPT.

The second stage in ReACC is to complete codes based on the context and the retrieved codes. We simply put the retrieved code before the context and concat them as inputs. 

Navigate to the `gen` folder. We adapt the code completion scripts in [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/CodeCompletion-line). We modify the script `dataset.py` to include similar codes as input. Run `run_lm.py` to evaluate your fine-tuned model.

```bash
export CUDA_VISIBLE_DEVICES=0
LANG=python
DATADIR=dataset/py150
LITFILE=${DATADIR}/literals.json
OUTPUTDIR=save/py150
PRETRAINDIR=py150-ckpt

LOADFILE=${DATADIR}/train_split
RESFILE=search_res.pkl
SAVEFILE=prediction.txt

python -u run_lm.py \
        --data_dir=$DATADIR \
        --lit_file=$LITFILE \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --load_file_name=$LOADFILE \
        --search_res=$RESFILE \
        --save_name=$SAVEFILE \
        --model_type=gpt2 \
        --block_size=1024 \
        --eval_line \
        --logging_steps=100 \
        --seed=42 
```


## Zero-shot code clone detection
In order to evaluate the effectiveness of the code-to-code retrieval module in ReACC, 
we perform code clone detection task which aims to retrieve semantic equivalent programs.

We extract the evaluation dataset from CodeNet, the same as in [UniXcoder paper](https://arxiv.org/abs/2203.03850).
The dataset can be downloaded from [here](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder/downstream-tasks/zero-shot-search/dataset)

Run the `codenet_test.py` to reproduce this experiment.
```bash
DATADIR=CodeNet
PRETRAINDIR=microsoft/reacc-py-retriever
 
python -u codenet_test.py \
        --data_dir=$DATADIR \
        --pretrained_dir=$PRETRAINDIR \
        --lang=python \
        --num_vec=8 \
        --cut \
        --block_size=512 \
        --per_gpu_eval_batch_size=64 \
        --logging_steps=100 \
        --seed=614 
```

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE) license.

## Reference
If you use this code or ReACC, please consider citing us.
<pre><code>@article{lu2022reacc,
  title={ReACC: A Retrieval-Augmented Code Completion Framework},
  author={Lu, Shuai and Duan, Nan and Han, Hojae and Guo, Daya and Hwang, Seung-won and Svyatkovskiy, Alexey},
  journal={arXiv preprint arXiv:2203.07722},
  year={2022}
}</code></pre>
