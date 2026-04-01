# Dual-Match Multi-Order Hash Embedding + CUDA Graph Microstep Replay (M34b)

This folder contains an aggressive systems variant of Dual-Match Multi-Order
Hash Embedding. Internally I refer to this branch as **M34b**, where the main
change relative to M34a is CUDA Graph replay at the microstep level.

This folder is not a record reproduction. It is a working research branch for:

- collision-robust n-gram hash embeddings
- compatibility with the XSA + Parallel Muon + GPTQ stack
- aggressive static-shape training experiments
- early 1x H100 proxy validation before 8x H100 runs

## What Is Preserved

The following parts are intentionally kept from the source stack:

- 11L x 512 backbone
- XSA on all 11 layers
- SmearGate
- VE / partial RoPE / LN scale
- Parameter Banking + Parallel Muon
- EMA + tight SWA
- AR self-generated GPTQ path
- selective pruning + LZMA compression

## What Is Changed

The old single-table local hash branch is replaced by Dual-Match Multi-Order
Hash Embedding, with the following design:

- `2/3/4-gram` are all enabled through `NGRAM_ORDERS`
- each order owns its own embedding region
- hashes within the same order share that order's embedding region
- multiple hash functions are still used to provide multiple candidate views
- candidate filtering uses a sigmoid `q * h` gate rather than softmax attention
- gate bias is shared per order, not per hash slot

In short:

- `order` controls parameter space
- `hash` controls access pattern
- the gate decides whether a candidate is semantically usable

## Current Module Shape

The current public interface is fully `NGRAM_*` based:

- `NGRAM_BASE_VOCAB_SIZE`
- `NGRAM_DIM`
- `NGRAM_ORDERS`
- `NGRAM_VOCAB_SIZES`
- `NGRAM_NUM_HASHES`
- `NGRAM_INIT_STD`

There is no legacy `BIGRAM_*` compatibility layer in this experiment copy.

## Current Best Proxy Result

Current notable 1x H100 10-minute proxy run:

- log: [log0401.txt](./log0401.txt)
- train wallclock: ~600s
- step time: ~217.9 ms
- pre-quant val BPB: `1.2374`
- int6 roundtrip val BPB: `1.24643762`
- int6 sliding-window val BPB: `1.22250731`
- total submission size: `13,580,697` bytes

This result is interesting because:

- the compressed artifact is well below 16MB
- the model appears not to be compression-bound yet
- the Dual-Match module is already compatible with the legal GPTQ pipeline

This is still a 1x H100 proxy result, not an 8x H100 record-equivalent run.

## Static Graph Variant

M34b adds a narrow CUDA Graph capture boundary around a single training
microstep:

- fixed-shape `x/y` staging buffers
- one captured forward + backward replay per microstep
- optimizer steps, `Muon`, and distributed communication remain eager

This is intentionally more aggressive than `torch.compile(..., fullgraph=True)`
alone, but still much safer than trying to graph the entire optimizer step.

## Run Commands

### 1x H100 Fair Proxy

Use `GRAD_ACCUM_STEPS=1` for fair single-GPU throughput checks.

```bash
RUN_ID=m34b_static_graph_1xh100 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=262144 \
GRAD_ACCUM_STEPS=1 \
VAL_BATCH_SIZE=262144 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=25 \
MAX_WALLCLOCK_SECONDS=600 \
NGRAM_BASE_VOCAB_SIZE=3072 \
NGRAM_DIM=112 \
NGRAM_ORDERS=2,3,4 \
NGRAM_VOCAB_SIZES=3072,1536,768 \
NGRAM_NUM_HASHES=2,2,2 \
NGRAM_INIT_STD=0.005 \
WARMDOWN_ITERS=4000 \
CUDAGRAPH_MICROSTEP=1 \
CUDAGRAPH_WARMUP_ITERS=3 \
TRAIN_COMPILE_MODE=reduce-overhead \
EVAL_COMPILE_MODE=default \
LATE_QAT_THRESHOLD=0 \
TARGET_MB=15.9 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### 8x H100 Starting Point

Suggested first 8x H100 config:

```bash
NGRAM_BASE_VOCAB_SIZE=3072 \
NGRAM_DIM=112 \
NGRAM_ORDERS=2,3,4 \
NGRAM_VOCAB_SIZES=3072,1536,768 \
NGRAM_NUM_HASHES=2,2,2 \
WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 \
GRAD_ACCUM_STEPS=1 \
TRAIN_BATCH_TOKENS=786432 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This is only a starting point. Batch sizing for 8x H100 is still an open
systems question for this branch.

## GPTQ Notes

This experiment branch keeps the legal AR self-generated GPTQ path, but the implementation
has already been modified to reduce obvious post-training overhead:

- autoregressive calibration generation now uses an incremental cached decode path
- Hessian collection now supports batched token sequences instead of one sequence
  per forward

This area is still under active optimization.

## Known Caveats

- TTT is not enabled in this branch.
- There is no separate "test-time ngram adaptation" path. The n-gram branch is
  part of the normal forward pass in both training and eval.
- The 1x H100 proxy can indicate early quality and systems health, but it does
  not fully predict 8x H100 behavior.
- Model size selection is still unresolved: this branch currently looks
  under-compressed rather than over-compressed on 1x H100.

## Design Philosophy

My main modeling belief is that a good neural architecture should not just stack popular tricks. It should introduce an inductive bias that still makes sense before we even talk about benchmarks.

In small hash tables, n-gram hash embeddings are naturally collision-heavy. This is manageable for bigrams, but it becomes much more severe for 3-gram and 4-gram features. At the same time, a 1024-token SentencePiece vocabulary often splits a single word into multiple subword pieces, so higher-order local patterns can still be important. This creates a tension: higher-order n-grams are potentially useful, but they are also much noisier under aggressive hashing.

My answer is to require two matches before an n-gram feature enters the residual stream: a hash match and a semantic match. The hash function retrieves candidate embeddings, but the model should still ask whether a candidate is actually compatible with the current token state. That is the motivation behind Dual-Match Multi-Order Hash Embedding (M34a): a multi-order hash module that filters candidates instead of trusting every hash hit equally.

### Works That Inspired Me

1. Gravity Tokenizer  
   https://github.com/dcrow85/Avalanche/blob/b0fee47daa91438c68e5e02493efb0bae0341484/gravity-tokenizer/submission/README.md  
   This reinforced the idea that some local token combinations are much more important than others.

2. DeepSeek Engram  
   https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/Engram_paper.pdf  
   This inspired the use of dot-product-style semantic filtering on top of hash retrieval.

3. Meituan-Longcat  
   https://arxiv.org/pdf/2601.21204  
   This showed that multiple hash views are a promising way to reduce collision damage, although I wanted a more selective mechanism than simple averaging.

### Engineering Decisions

1. Same order, same table  
   Hash functions within the same n-gram order share one embedding region. This improves parameter efficiency and reduces the chance of having many underused slots.

2. Sigmoid gate instead of softmax  
   If all retrieved candidates are noise, the model should be able to suppress all of them. A sigmoid gate supports that behavior more naturally than a softmax mixture.

3. Two hash functions instead of four  
   I originally used four hashes per order to reduce collisions, but the extra retrieval and gating cost became too IO-heavy. Reducing this to two kept most of the quality while making the system much cheaper to run.



## About Me

Hi! I'm Yicheng Xia, a first-year student at the University of Toronto studying Mathematical Applications in Economics and Finance. I’m especially interested in thinking about unconventional neural network architectures, compression ideas, and systems-aware training tricks.

I'm actively seeking internship, research, and hackathon opportunities. Feel free to connect with me on LinkedIn: www.linkedin.com/in/yicheng-xia-2576b63a9

## Lineage

This experiment branch is derived from the legal SOTA family around:

- PR #549
- PR #609
- the 2026-03-25 `ValCalib_GPTQ_XSA_BigramHash3072` stack

The goal is not to copy the full record recipe blindly, but to test whether
Dual-Match Multi-Order Hash Embedding (M34a) can survive inside a strong modern
stack without losing training speed or legal quantization compatibility.

## Requirements

This script expects the same core environment assumptions as the copied SOTA
stack:

- Hopper-class GPU for the FlashAttention 3 path
- PyTorch + CUDA environment compatible with `flash_attn_interface`
- `sentencepiece`
- `zstandard` optional, otherwise compression falls back to `zlib`
